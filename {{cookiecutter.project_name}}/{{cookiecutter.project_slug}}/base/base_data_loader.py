from typing import Callable

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, collate_fn: Callable = default_collate
    ) -> None:
        """
        Base class for all data loaders
        Args:
            dataset: Dataset to load data from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of subprocesses to use for data loading
            collate_fn: Function to merge a list of samples to form a mini-batch
        """
        self.batch_size = batch_size
        super().__init__(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)
