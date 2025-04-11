import torch


def mse_metric(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error metric. The metric is calculated as the mean squared error between the output and target.

    Arguments:
        output: tensor, predicted values
        target: tensor, target values

    Returns:
        mse: float, mean squared error
    """
    mse = torch.mean((output - target) ** 2)
    return mse
