import torch
import torch.nn as nn
from collections.abc import Mapping

from mlcore.core import ComponentType, Task, COMPONENTS


@COMPONENTS.register(
    name="ComplexLoss",
    ctype=ComponentType.LOSS,
    supported_tasks=[Task.CLASSIFICATION, Task.SEMANTIC_SEGMENTATION],
    description="Computes several loss with provided weights factor",
)
class WeightedLoss(nn.Module):
    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        **losses: dict[str, nn.Module],
    ) -> None:
        super().__init__()

        if not losses:
            raise ValueError("ComplexLoss requires at least one child loss")

        for name, loss in losses.items():
            if not isinstance(loss, nn.Module):
                msg = "`ComplexLoss` children losses must be nn.Module instances\n"
                msg += f"Provided loss: {name} class {type(loss)}"
                raise ValueError(msg)

        self.losses = nn.ModuleDict(losses)

        if weights is None:
            weights = {name: 1.0 for name in self.losses}
        else:
            unknown = set(weights).difference(self.losses)
            missing = set(self.losses).difference(weights)
            if unknown or missing:
                details: list[str] = []
                if unknown:
                    details.append(f"unknown weights: {', '.join(sorted(unknown))}")
                if missing:
                    details.append(f"missing weights: {', '.join(sorted(missing))}")
                raise ValueError(f"ComplexLoss weights mismatch ({'; '.join(details)})")

        self.weights = {name: float(weight) for name, weight in weights.items()}

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = inputs.new_tensor(0.0)
        for name, loss_fn in self.losses.items():
            loss = loss_fn(inputs, targets)
            total_loss = total_loss + self.weights[name] * loss
        return total_loss

    def extra_repr(self) -> str:
        return " \n".join([f"{key} {value}" for key, value in self.weights.items()])
