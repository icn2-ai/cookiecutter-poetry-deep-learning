import torch
from torch.utils.data import Dataset


class RandomVectorDataset(Dataset):
    def __init__(self, num_samples: int, vector_size: int = 128) -> None:
        """
        Dataset that generates random vectors of given size.
        Args:
            num_samples (int): Number of samples in the dataset.
            vector_size (int): Size of each random vector.
        """
        self.num_samples = num_samples
        self.vector_size = vector_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        # Return the random vector and its corresponding target
        return torch.randn(self.vector_size), torch.randn(self.vector_size)
