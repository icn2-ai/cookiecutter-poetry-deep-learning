import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self) -> None:
        """
        Mean squared error loss. The loss is calculated as the mean squared error between the output and target.

        Arguments:
            output: tensor, predicted values
            target: tensor, target values

        Returns:
            loss: float, mean squared error loss
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(output, target, reduction="mean")
