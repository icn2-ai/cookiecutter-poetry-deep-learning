import torch
import torch.nn as nn

from {{cookiecutter.project_slug}}.base.base_model import BaseModel


class IdentityModel(BaseModel):
    def __init__(self) -> None:
        """
        Identity model. The model is used to test the training loop and the data loading.
        """
        super().__init__()
        self.dummy_param = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * self.dummy_param
