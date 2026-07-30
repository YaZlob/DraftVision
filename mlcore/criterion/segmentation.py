import torch
import torch.nn as nn
import torch.nn.functional as F
from mlcore.core import COMPONENTS, ComponentType, Task


@COMPONENTS.register(
    "TverskyLoss",
    ctype=ComponentType.LOSS,
    supported_tasks=Task.SEMANTIC_SEGMENTATION,
    description="Tversky loss for semantic semgentation",
)
class Tversky(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, ignore_index=255):
        """
        Tversky Loss for semantic segmentation.
        Args:
            alpha (float): Weight of false positives
            beta (float): Weight of false negatives
        """
        super().__init__()

        assert alpha + beta == 1

        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, pred, target) -> torch.Tensor:
        assert target.dim() == 3 and pred.dim() == 4

        B, C, H, W = pred.shape
        prob = F.softmax(pred, dim=1)
        # fmt: off
        prob = prob.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        target = target.reshape(-1)                     # [B*H*W]
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            prob = prob[valid_mask]
            target = target[valid_mask]
            if target.size(0) == 0:
                return torch.tensor(0.0, device=pred.device)
        
        one_hot = F.one_hot(target, num_classes=C).to(prob.dtype)   # [N, C]
        tp = (prob * one_hot).sum(dim=0)                            # [C]
        fp = (prob * (1 - one_hot)).sum(dim=0)
        fn = ((1 - prob) * one_hot).sum(dim=0)
        # fmt: on
        numerator = tp + 1e-6
        denominator = tp + self.alpha * fp + self.beta * fn + 1e-6
        tversky_per_class = numerator / denominator  # [C]
        tversky_loss = 1.0 - tversky_per_class.mean()
        return tversky_loss
