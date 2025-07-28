from copy import deepcopy
from pathlib import Path
from typing import Sized, Tuple

import torch.multiprocessing
from numpy import float32
from numpy import genfromtxt, int32
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

PATH_DATASETS = Path("../data")
PATH_IRIS = PATH_DATASETS / "iris.txt"
PATH_WINE = PATH_DATASETS / "wine.txt"


class CustomDataset(Dataset[Tuple[torch.Tensor, float32]], Sized):
    def __init__(self, path: Path) -> None:
        data = genfromtxt(path, delimiter=',')
        self.x = StandardScaler().fit_transform(data[:, 1:])
        self.y = data[:, 0].astype(int32)
        self._classes = list(set(self.y))

    def __getitem__(self, index: int) -> (torch.Tensor, float32):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def classes(self) -> list[int]:
        return deepcopy(self._classes)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_features(self) -> int:
        return self.x.shape[1]


class IrisDataset(CustomDataset):
    def __init__(self, data: Path = PATH_IRIS) -> None:
        super().__init__(data)


class IrisDataset2D(CustomDataset):
    def __init__(self, data: Path = PATH_IRIS) -> None:
        super().__init__(data)
        data = genfromtxt(data, delimiter=",")[:100]
        self.x = StandardScaler().fit_transform(data[:, 1:])
        self.y = data[:, 0].astype(int32)
        self._classes = list(set(self.y))


class WineDataset(CustomDataset):
    def __init__(self, data: Path = PATH_WINE) -> None:
        super().__init__(data)


class WineDataset2D(CustomDataset):
    def __init__(self, data: Path = PATH_WINE) -> None:
        super().__init__(data)
        data = genfromtxt(data, delimiter=",")[:130]
        self.x = StandardScaler().fit_transform(data[:, 1:])
        self.y = data[:, 0].astype(int32)
        self._classes = list(set(self.y))
