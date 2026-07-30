from collections.abc import Callable
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Base map-style dataset with an optional joint image/target transform."""

    def __init__(self, transforms: Callable | None = None):
        self.transforms = transforms

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def load_item(self, index):
        raise NotImplementedError(
            "Please implement this function to return item before `transforms`."
        )

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1
