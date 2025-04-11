from typing import Any

from torch.utils.data.dataloader import default_collate

from {{cookiecutter.project_slug}}.base.base_data_loader import BaseDataLoader

from .data_sets import RandomVectorDataset


class RandomVectorDataLoader(BaseDataLoader):
    def __init__(
        self,
        num_samples: int,
        vector_size: int,
        batch_size: int = 32,
        num_workers: int = 2,
        shuffle: bool = True,
        collate_fn: Any = default_collate,
    ) -> None:
        """
        DataLoader for random vector dataset.

        Args:
            num_samples (int): Number of samples in the dataset.
            vector_size (int): Size of each random vector.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            shuffle (bool): Whether to shuffle the data at every epoch.
            collate_fn (callable): Function to merge a list of samples to form a mini-batch.
        """
        dataset = RandomVectorDataset(num_samples=num_samples, vector_size=vector_size)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
