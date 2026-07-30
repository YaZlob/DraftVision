import torch
import torch.nn as nn
import torch.nn.functional as F
from mlcore.core import COMPONENTS, ComponentType, Task


@COMPONENTS.register(
    name="CrossEntropyLoss",
    ctype=ComponentType.LOSS,
    supported_tasks=[Task.CLASSIFICATION, Task.SEMANTIC_SEGMENTATION],
    provider="torch",
    description="Default pytorch cross entropy loss",
)
def build_ce(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    if weight is not None and not isinstance(weight, torch.Tensor):
        raise ValueError("If provided `weight` must be torch.Tensor")
    return nn.CrossEntropyLoss(
        weight, size_average, ignore_index, reduce, reduction, label_smoothing
    )


@COMPONENTS.register(
    name="FocalLoss",
    ctype=ComponentType.LOSS,
    supported_tasks=[Task.CLASSIFICATION, Task.SEMANTIC_SEGMENTATION],
    description="Focal loss implementation",
)
class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | list | None = None,
        gamma: float = 2.0,
        reduction: str | None = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()

        if reduction:
            if not isinstance(reduction, str) or not reduction in ["sum", "mean"]:
                raise ValueError(
                    f"If provided reduction must be `mean` or `sum`, got {reduction}"
                )
            self.reduction = reduction
        else:
            self.reduction = None

        if not isinstance(gamma, float):
            raise ValueError("FocalLoss gamma must be float")

        self.gamma = gamma
        self.ignore_index = ignore_index

        if weight is not None:
            if isinstance(weight, (list, tuple)):
                weight = torch.Tensor(weight)
            elif not isinstance(weight, torch.Tensor):
                msg = f"FocalLoss weight expected as torch.Tensor | list |None, got {type(weight)}"
                raise ValueError(msg)
        self.alpha = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

            safe_targets = targets.clone()
            if self.ignore_index is not None:
                safe_targets[targets == self.ignore_index] = 0

            # Возьми из alpha элементы по индексу 0, 2, 1, 0
            # alpha = tensor([0.2, 0.5, 1.0])
            # targets = tensor([[[0, 2], [1, 0]]])
            # alpha.gather(0, safe_targets.view(-1)) -> tensor([0.2, 1.0, 0.5, 0.2])

            at = alpha.gather(0, safe_targets.view(-1)).view(targets.shape)
            focal_loss = at * focal_loss

        if self.reduction == "mean":
            if self.ignore_index is not None:
                valid_mask = targets != self.ignore_index
                return focal_loss[valid_mask].mean()
            else:
                return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":

    weigth = torch.empty(5, dtype=torch.float32).random_(10)
    input = torch.randn(5, 5, requires_grad=True)
    target = torch.empty(5, dtype=torch.long).random_(5)

    target[1] = 100
    fl = COMPONENTS.build(
        "FocalLoss",
        ctype=ComponentType.LOSS,
        task=Task.CLASSIFICATION,
        params={"alpha": weigth, "ignore_index": 100},
    )

    out = fl(input, target)
    print(out)
