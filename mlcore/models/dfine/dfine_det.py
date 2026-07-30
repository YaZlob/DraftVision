import torch
import torch.nn as nn
from mlcore.core import COMPONENTS, ComponentType, Task


@COMPONENTS.register(
    name="DfineDetecor",
    ctype=ComponentType.MODEL,
    supported_tasks=Task.OBJECT_DETECTION,
    provider="https://github.com/Peterande/D-FINE",
    description="Dfine object detector with little adaptation",
)
class DfineDet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        postprocessor: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        self.postprocessor = postprocessor

    def forward(self, x, *, targets=None, orig_sizes=None) -> dict[str, torch.Tensor]:
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        if self.training:
            return x
        return self.postprocessor(x, orig_sizes)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
