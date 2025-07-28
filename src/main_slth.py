import random
from os import makedirs
from time import time
from typing import Literal, List, Tuple

import numpy as np
import pandas as pd
import pennylane
import torch
from torchmetrics import Accuracy

from config import get_config, Dataset
from main_wlth import get_dataloaders
from models import Model, NN, VQC, BVQC

N_GENERATIONS = 75
N_INDIVIDUALS = 25

N_AFTER_SELECTION = int(0.333 * N_INDIVIDUALS)
N_AFTER_RECOMBINATION = int(0.666 * N_INDIVIDUALS)
N_AFTER_MUTATION = int(0.95 * N_INDIVIDUALS)
MUTATION_RATE = 0.35
N_TO_MIGRATE = N_INDIVIDUALS - N_AFTER_MUTATION

REFERENCE_MODEL: BVQC | VQC | NN | None = None


def set_random_seed(seed: int) -> None:
    print(f'\nSetting random seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    pennylane.numpy.random.seed(seed)
    random.seed(seed)


def get_new_model(config, dataset):
    new_model: BVQC | VQC | NN = Model.get_instance(config, dataset)
    new_model.load_weights_copy(REFERENCE_MODEL.get_weights_copy())
    return new_model


def main():
    global REFERENCE_MODEL

    df_data = []

    for random_seed in range(10):
        start_time = time()

        set_random_seed(random_seed)

        config = get_config()
        dataset = Dataset.get_instance(config.dataset)
        REFERENCE_MODEL = Model.get_instance(config, dataset)

        simulation(df_data, random_seed, config, dataset)

        print(f'Took {(time() - start_time) / 60:.1f} minutes')

    config = get_config()
    folder_name = 'dataframes/slth'
    makedirs(folder_name, exist_ok=True)
    file_name = f'{config.model}_{config.dataset}_{N_GENERATIONS}x{N_INDIVIDUALS}.csv'

    df = pd.DataFrame(df_data)
    df.to_csv(f'{folder_name}/{file_name}', index=False)


def simulation(dataframe_data: list, random_seed: int, config, dataset):
    train_dl, valid_dl, test_dl = get_dataloaders(dataset, config.batch_size)

    individuals = mutate_individuals([get_new_model(config, dataset)], config, dataset, 0.5)

    for i_generation in range(N_GENERATIONS):
        print(f'Gen {i_generation + 1:2d}/{N_GENERATIONS:2d}...')
        fitnesses = test_individuals(individuals, train_dl, config, dataset)
        save_individuals(dataframe_data, i_generation, fitnesses, random_seed)

        individuals = select_individuals(fitnesses)
        individuals = recombine_individuals(individuals, config, dataset)
        individuals = mutate_individuals(individuals, config=config, dataset=dataset)
        individuals = migrate_individuals(individuals, config, dataset)

    # measuring fitness after last generation
    fitnesses = test_individuals(individuals, train_dl, config, dataset)
    save_individuals(dataframe_data, N_GENERATIONS, fitnesses, random_seed)


def save_individuals(dataframe_data: list, i_generation: int, fitnesses, random_value: int):
    fitnesses = sorted(fitnesses, key=lambda x: x[1], reverse=False)

    for i_individual, (individual, accuracy) in enumerate(fitnesses):
        dataframe_data.append({
            'i_gen': i_generation,
            'i_individual': i_individual,
            'accuracy': accuracy,
            'sparsity': individual.remaining_weights_unrounded,
            'random_value': random_value,
        })


def test_individuals(individuals: List, dataloader, config, dataset) -> List[Tuple]:
    def evaluate_individual(individual, dataloader, num_classes) -> float:
        task: Literal["binary", "multiclass"] = "binary" if config.model == Model.BVQC_NAME.value else "multiclass"
        accuracy = Accuracy(threshold=0.5, num_classes=num_classes, task=task)

        for x, y in dataloader:
            y_pred = individual(x)
            accuracy.update(y_pred, y.long())

        return accuracy.compute().item()

    return [(individual, evaluate_individual(individual, dataloader, dataset.n_classes)) for individual in individuals]


def select_individuals(fitnesses: list[tuple]) -> list:
    def get_weights(model):
        return model.get_weights_mask_copy()

    def are_weights_equal(weights1, weights2):
        if isinstance(weights1, list):
            return all((w1 == w2).all() for w1, w2 in zip(weights1, weights2))
        return (weights1 == weights2).all()

    clean_fitnesses = [fitnesses[0]]

    for index_outer, (model_outer, _) in enumerate(fitnesses[:-2]):
        model_inner, accuracy_inner = fitnesses[index_outer + 1]

        if are_weights_equal(get_weights(model_outer), get_weights(model_inner)):
            continue

        clean_fitnesses.append((model_inner, accuracy_inner))

    sorted_individuals = sorted(clean_fitnesses, key=lambda x: x[1], reverse=True)
    return [individual for individual, _ in sorted_individuals][:N_AFTER_SELECTION]


def recombine_individuals(individuals: list, config, dataset) -> list:
    new_individuals = []

    # Determine if the weights mask is a tensor or a list of tensors
    first_mask = individuals[0].get_weights_mask_copy()
    is_list_of_tensors = isinstance(first_mask, list)

    if is_list_of_tensors:
        shape = [tensor.shape for tensor in first_mask]
        length_half = [len(t.flatten()) // 2 for t in first_mask]
    else:
        shape = first_mask.shape
        length_half = len(first_mask.flatten()) // 2

    while len(individuals + new_individuals) < N_AFTER_RECOMBINATION:
        first_individual_index = random.randint(0, len(individuals) - 1)
        second_individual_index = random.randint(0, len(individuals) - 1)
        if first_individual_index == second_individual_index:
            second_individual_index = (second_individual_index + 1) % len(individuals)

        first_individual = individuals[first_individual_index]
        second_individual = individuals[second_individual_index]

        first_mask = first_individual.get_weights_mask_copy()
        second_mask = second_individual.get_weights_mask_copy()

        if is_list_of_tensors:
            new_weights_mask = []
            for i in range(len(first_mask)):
                first_half = first_mask[i].flatten()[:length_half[i]].tolist()
                second_half = second_mask[i].flatten()[length_half[i]:].tolist()
                new_tensor = torch.tensor(first_half + second_half).reshape(shape[i])
                new_weights_mask.append(new_tensor)
        else:
            first_half = first_mask.flatten()[:length_half].tolist()
            second_half = second_mask.flatten()[length_half:].tolist()
            new_weights_mask = torch.tensor(first_half + second_half).reshape(shape)

        new_model = get_new_model(config, dataset)
        new_model.prune_custom(new_weights_mask)
        new_individuals.append(new_model)

    return individuals + new_individuals


def mutate_individuals(individuals: list,
                       config, dataset,
                       mutation_rate: float = MUTATION_RATE,
                       n_after_mutation: int = N_AFTER_MUTATION) -> list:
    new_individuals = []

    while len(individuals + new_individuals) < n_after_mutation:
        individual_mask = random.choice(individuals).get_weights_mask_copy()
        new_individual = get_new_model(config, dataset)
        new_individual.prune_custom(individual_mask)
        new_individual.mutate(mutation_rate)
        new_individuals.append(new_individual)

    return individuals + new_individuals


def migrate_individuals(individuals: list, config, dataset) -> list:
    return individuals + mutate_individuals([get_new_model(config, dataset)],
                                            config, dataset,
                                            mutation_rate=0.5, n_after_mutation=N_TO_MIGRATE)


if __name__ == '__main__':
    main()
