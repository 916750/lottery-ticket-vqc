from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List

import pennylane as qml
import torch
from numpy.random import uniform
from torch import Tensor, nn
from torch.nn.utils import prune


@dataclass
class MaskInfo:
    n_pruned: int
    n_unpruned: int
    n_total: int


class NN(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_input, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_output),
            nn.Softmax(),
        )

        self.linears = [module
                        for module in self.model._modules.values()
                        if isinstance(module, nn.Linear)]

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.float())

    @property
    def remaining_weights_unrounded(self):
        def is_pruned():
            return any('mask' in key for key in state_dict)

        def get_unpruned_and_total():
            unpruned, total = 0, 0
            for key, value in state_dict.items():
                if "mask" in key:
                    unpruned += torch.sum(value != 0).item()
                    total += value.numel()
            return unpruned, total

        state_dict = self.model.state_dict()

        if not is_pruned():
            return 100.0

        unpruned, total = get_unpruned_and_total()
        remaining_weights_unrounded = unpruned / total * 100
        return remaining_weights_unrounded

    @property
    def remaining_weights(self):
        return round(self.remaining_weights_unrounded, 1)

    def prune(self, pruning_rate):
        for module in self.linears:
            prune.l1_unstructured(module, "weight", float(pruning_rate))

    def get_weights_copy(self):
        weights = []
        for module in self.linears:
            state_dict = deepcopy(module.state_dict())
            key = 'weight' if 'weight' in state_dict else 'weight_orig'
            weights.append(state_dict[key])
        return weights

    def get_weights_mask_copy(self):
        weights = []
        for module in self.linears:
            state_dict = deepcopy(module.state_dict())

            if 'weight_mask' in state_dict:
                weights.append(state_dict['weight_mask'])
            else:
                weights.append(torch.ones_like(state_dict['weight']))

        return weights

    def load_weights_copy(self, weights_copies):
        for module, new_weights in zip(self.linears, weights_copies):
            state_dict = module.state_dict()
            key = 'weight' if 'weight' in state_dict else 'weight_orig'
            state_dict[key] = deepcopy(new_weights)

    def get_mask_information(self) -> MaskInfo:
        unpruned, total = 0, 0

        for key, value in self.model.state_dict().items():
            if "mask" in key:
                unpruned += torch.sum(value != 0).item()
                total += value.numel()

        return MaskInfo(total - unpruned, unpruned, total)

    def mutate(self, mutation_rate: float):
        for module in self.linears:
            if not hasattr(module, 'weight_mask'):
                prune.l1_unstructured(module, 'weight', .0)

            mask = module.weight_mask.flatten().bool()
            num_to_flip = int(mutation_rate * len(mask))
            indices = torch.randperm(mask.numel())[:num_to_flip]
            mask[indices] = ~mask[indices]  # Flip the selected elements
            module.weight_mask = mask.reshape(module.weight_mask.shape).float()

    def prune_custom(self, new_weights_mask):
        for module, mask in zip(self.linears, new_weights_mask):
            prune.custom_from_mask(module, 'weight', mask)


class NNSimple(NN):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__(n_input, n_output)

        self.model = nn.Sequential(
            nn.Linear(n_input, 24),
            nn.ReLU(),
            nn.Linear(24, n_output),
            nn.Softmax(),
        )

        self.linears = [module
                        for module in self.model._modules.values()
                        if isinstance(module, nn.Linear)]


class VQC(nn.Module):
    def __init__(self,
                 classes: List[int],
                 num_qubits: int,
                 num_layers: int,
                 shall_data_reupload: bool,
                 uniform_range: float
                 ) -> None:
        super().__init__()
        self.device = qml.device("default.qubit", wires=num_qubits)
        self.classes = classes

        def circuit(layer_weights_list: Tensor, x: Tensor):
            if shall_data_reupload:
                for layer_weights in layer_weights_list:
                    qml.AngleEmbedding(x, wires=range(num_qubits))
                    qml.StronglyEntanglingLayers(layer_weights.unsqueeze(0), wires=range(num_qubits))
            else:
                qml.AngleEmbedding(x, wires=range(num_qubits))
                qml.StronglyEntanglingLayers(layer_weights_list, wires=range(num_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(len(self.classes))]

        self.qnode = qml.QNode(circuit, self.device, interface="torch")

        weights_count = num_layers * num_qubits * 3
        weights_shape = (num_layers, num_qubits, 3)
        weights = Tensor(uniform(-uniform_range, uniform_range, weights_count).reshape(weights_shape))
        self.weights = nn.Parameter(weights, requires_grad=True)

        self.bias = nn.Parameter(torch.zeros(len(self.classes)), requires_grad=True)
        self.output_scale_factor = nn.Parameter(torch.ones(len(self.classes)), requires_grad=True)

    @property
    def remaining_weights_unrounded(self):
        if not hasattr(self, 'weights_mask'):
            return 100.0

        count_unequal_zero = torch.sum(self.weights_mask != 0).item()
        total = len(self.weights.flatten())
        remaining_weights = count_unequal_zero / total * 100
        return remaining_weights

    @property
    def remaining_weights(self):
        return round(self.remaining_weights_unrounded, 1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack(
            [nn.Softmax()((qml.math.stack(self.qnode(self.weights, state)) + self.bias) * self.output_scale_factor)
             for state in x])

    def prune(self, pruning_rate):
        prune.l1_unstructured(self, 'weights', float(pruning_rate))

    def get_weights_copy(self):
        state_dict_copy = deepcopy(self.state_dict())
        key = 'weights' if 'weights' in state_dict_copy else 'weights_orig'
        return state_dict_copy[key]

    def get_weights_mask_copy(self):
        state_dict_copy = deepcopy(self.state_dict())
        return state_dict_copy.get('weights_mask', torch.ones_like(self.weights))

    def load_weights_copy(self, weights_copy):
        state_dict = self.state_dict()
        key = 'weights' if 'weights' in state_dict else 'weights_orig'
        state_dict[key] = deepcopy(weights_copy)

    def prune_custom(self, new_weights_mask):
        prune.custom_from_mask(self, 'weights', new_weights_mask)

    def get_mask_information(self) -> MaskInfo:
        weights_mask = self.state_dict()['weights_mask']

        n_weights = self.weights.numel()
        n_pruned_weights = (torch.sum(weights_mask == 0).item()
                            if weights_mask is not None else
                            0)
        n_unpruned_weights = n_weights - n_pruned_weights

        return MaskInfo(n_pruned_weights, n_unpruned_weights, n_weights)

    def draw(self, x: Tensor):
        print(qml.draw(self.qnode)(self.weights, x))

    def mutate(self, mutation_rate: float):
        if not hasattr(self, 'weights_mask'):
            prune.l1_unstructured(self, 'weights', .0)

        mask = self.weights_mask.flatten().bool()
        num_to_flip = int(mutation_rate * len(mask))
        indices = torch.randperm(mask.numel())[:num_to_flip]
        mask[indices] = ~mask[indices]  # Flip the selected elements
        self.weights_mask = mask.reshape(self.weights_mask.shape).float()


class BVQC(VQC):
    def __init__(self,
                 classes: List[int],
                 num_qubits: int,
                 num_layers: int,
                 shall_data_reupload: bool,
                 uniform_range: float
                 ) -> None:
        super().__init__(classes, num_qubits, num_layers, shall_data_reupload, uniform_range)

        def circuit(layer_weights_list: Tensor, x: Tensor):
            if shall_data_reupload:
                for layer_weights in layer_weights_list:
                    qml.AngleEmbedding(x, wires=range(num_qubits))
                    qml.StronglyEntanglingLayers(layer_weights.unsqueeze(0), wires=range(num_qubits))
            else:
                qml.AngleEmbedding(x, wires=range(num_qubits))
                qml.StronglyEntanglingLayers(layer_weights_list, wires=range(num_qubits))

            return qml.expval(qml.PauliZ(0))

        self.qnode = qml.QNode(circuit, self.device, interface="torch")

        self.bias = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.output_scale_factor = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([(self.qnode(self.weights, state) + self.bias) * self.output_scale_factor
                            for state in x])


class Model(Enum):
    BVQC_NAME = "BVQC"
    VQC_NAME = "VQC"
    NNS_NAME = "SNN"
    NN_NAME = "NN"

    @staticmethod
    def get_names():
        return [
            Model.BVQC_NAME.value,
            Model.VQC_NAME.value,
            Model.NNS_NAME.value,
            Model.NN_NAME.value
        ]

    @staticmethod
    def get_instance(config, dataset) -> BVQC | VQC | NN | NNSimple:
        match config.model:
            case Model.VQC_NAME.value:
                return VQC(
                    dataset.classes,
                    max(dataset.n_classes, dataset.n_features),
                    config.num_layers,
                    config.shall_data_reupload,
                    config.uniform_range
                )

            case Model.BVQC_NAME.value:
                if dataset.n_classes > 2:
                    print(f'The BVQC only works with 2 classes, but {dataset.n_classes} were given.')
                    exit(0)

                return BVQC(
                    dataset.classes,
                    max(dataset.n_classes, dataset.n_features),
                    config.num_layers,
                    config.shall_data_reupload,
                    config.uniform_range
                )

            case Model.NN_NAME.value:
                return NN(dataset.n_features, dataset.n_classes)

            case Model.NNS_NAME.value:
                return NNSimple(dataset.n_features, dataset.n_classes)

            case _:
                raise NotImplementedError(f'Model {config.model} not supported.')
