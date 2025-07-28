import sys
from dataclasses import dataclass
from enum import Enum

from dataset import CustomDataset, IrisDataset, WineDataset, IrisDataset2D, WineDataset2D
from models import Model


class Dataset(Enum):
    IRIS2 = "2iris"
    IRIS = "3iris"
    WINE2 = "2wine"
    WINE = "3wine"

    @staticmethod
    def get_names():
        return [Dataset.IRIS2.value,
                Dataset.IRIS.value,
                Dataset.WINE.value,
                Dataset.WINE2.value]

    @staticmethod
    def get_instance(dataset: str) -> CustomDataset:
        match dataset:
            case Dataset.IRIS2.value:
                return IrisDataset2D()
            case Dataset.IRIS.value:
                return IrisDataset()
            case Dataset.WINE.value:
                return WineDataset()
            case Dataset.WINE2.value:
                return WineDataset2D()
            case _:
                raise ValueError(f"No such dataset: {dataset}!")


@dataclass
class Config:
    model: str
    dataset: str

    batch_size: int
    lr: float
    weight_decay: float

    num_layers: int | None = None
    shall_data_reupload: bool | None = None
    uniform_range: float | None = None

    epochs: int = 50
    seed_value: int = 0


CONFIG_2IRIS_BVQC = Config(model=Model.BVQC_NAME.value,
                           dataset=Dataset.IRIS2.value,
                           batch_size=16,
                           lr=0.0061690543775444456,
                           weight_decay=0.00011451748647630793,
                           num_layers=10,
                           shall_data_reupload=False,
                           uniform_range=1.7834073464102067)

CONFIG_2IRIS_VQC = Config(model=Model.VQC_NAME.value,
                          dataset=Dataset.IRIS2.value,
                          batch_size=11,
                          lr=0.14046032832955133,
                          weight_decay=0.0002004341948131265,
                          num_layers=15,
                          shall_data_reupload=False,
                          uniform_range=0.34099999999999997)

CONFIG_2IRIS_NN = Config(model=Model.NN_NAME.value,
                         dataset=Dataset.IRIS2.value,
                         batch_size=15,
                         lr=0.00527192555914151,
                         weight_decay=0.0007875828517008305)

CONFIG_2IRIS_NNS = Config(model=Model.NNS_NAME.value,
                          dataset=Dataset.IRIS2.value,
                          batch_size=10,
                          lr=0.018975394132933525,
                          weight_decay=0.0003795868684963181)

CONFIG_3IRIS_VQC = Config(model=Model.VQC_NAME.value,
                          dataset=Dataset.IRIS.value,
                          batch_size=12,
                          lr=0.027667465613327877,
                          weight_decay=0.00014188599748059832,
                          num_layers=16,
                          shall_data_reupload=False,
                          uniform_range=0.997)

CONFIG_3IRIS_NN = Config(model=Model.NN_NAME.value,
                         dataset=Dataset.IRIS.value,
                         batch_size=6,
                         lr=0.004769194015507301,
                         weight_decay=0.00022149501355195838)

CONFIG_3IRIS_NNS = Config(model=Model.NNS_NAME.value,
                          dataset=Dataset.IRIS.value,
                          batch_size=11,
                          lr=0.041287606449925456,
                          weight_decay=0.000114582943114774)

CONFIG_2WINE_BVQC = Config(model=Model.BVQC_NAME.value,
                           dataset=Dataset.WINE2.value,
                           batch_size=14,
                           lr=0.034116351302152584,
                           weight_decay=0.0004883276632230364,
                           num_layers=14,
                           shall_data_reupload=False,
                           uniform_range=0.7665926535897931)

CONFIG_2WINE_VQC = Config(model=Model.VQC_NAME.value,
                          dataset=Dataset.WINE2.value,
                          batch_size=13,
                          lr=0.04693603770646666,
                          weight_decay=0.00017400041959447874,
                          num_layers=9,
                          shall_data_reupload=False,
                          uniform_range=0.13840734641020713)

CONFIG_2WINE_NN = Config(model=Model.NN_NAME.value,
                         dataset=Dataset.WINE2.value,
                         batch_size=16,
                         lr=0.015332376848592805,
                         weight_decay=0.0003668485961234364)

CONFIG_2WINE_NNS = Config(model=Model.NNS_NAME.value,
                          dataset=Dataset.WINE2.value,
                          batch_size=10,
                          lr=0.0012576386169755418,
                          weight_decay=0.0007737550251708995)

CONFIG_WINE_VQC = Config(model=Model.VQC_NAME.value,
                         dataset=Dataset.WINE.value,
                         batch_size=13,
                         lr=0.05629217353567388,
                         weight_decay=0.00033968499286871637,
                         num_layers=16,
                         shall_data_reupload=False,
                         uniform_range=0.353)

CONFIG_WINE_NN = Config(model=Model.NN_NAME.value,
                        dataset=Dataset.WINE.value,
                        batch_size=9,
                        lr=0.008599114360067295,
                        weight_decay=0.00022736976170105946)

CONFIG_WINE_NNS = Config(model=Model.NNS_NAME.value,
                         dataset=Dataset.WINE.value,
                         batch_size=9,
                         lr=0.05744528221412398,
                         weight_decay=0.00010743993876395757)


def get_config() -> 'Config':
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]

    if model_name not in Model.get_names():
        raise ValueError(f"Model \'{model_name}\' not supported.")

    if model_name == Model.BVQC_NAME.value and dataset_name in ["3iris", "wine"]:
        print(f'The {dataset_name} dataset is not supported for the BVQC model.')
        exit(0)

    match (dataset_name, model_name):
        case ("2iris", "BVQC"):
            return CONFIG_2IRIS_BVQC
        case ("2iris", "VQC"):
            return CONFIG_2IRIS_VQC
        case ("2iris", "NN"):
            return CONFIG_2IRIS_NN
        case ("2iris", "SNN"):
            return CONFIG_2IRIS_NNS

        case ("3iris", "VQC"):
            return CONFIG_3IRIS_VQC
        case ("3iris", "NN"):
            return CONFIG_3IRIS_NN
        case ("3iris", "SNN"):
            return CONFIG_3IRIS_NNS

        case ("2wine", "BVQC"):
            return CONFIG_2WINE_BVQC
        case ("2wine", "VQC"):
            return CONFIG_2WINE_VQC
        case ("2wine", "NN"):
            return CONFIG_2WINE_NN
        case ("2wine", "SNN"):
            return CONFIG_2WINE_NNS

        case ("3wine", "VQC"):
            return CONFIG_WINE_VQC
        case ("3wine", "NN"):
            return CONFIG_WINE_NN
        case ("3wine", "SNN"):
            return CONFIG_WINE_NNS

        case _:
            raise ValueError(f'Combination of dataset \'{dataset_name}\' and model \'{model_name}\' not supported.')


class PruningTechnique(Enum):
    ONE_SHOT = 'ONE_SHOT'
    ITERATIVE = 'ITERATIVE'

    @staticmethod
    def get_instance(pruning_technique: str) -> 'PruningTechnique':
        match pruning_technique:
            case "ONE_SHOT":
                return PruningTechnique.ONE_SHOT
            case "ITERATIVE":
                return PruningTechnique.ITERATIVE
            case _:
                raise ValueError(f'Pruning technique \'{pruning_technique}\' does not exist.')
