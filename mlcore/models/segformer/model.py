import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping

from mlcore.core import COMPONENTS, ComponentType, Task

try:
    from transformers import AutoConfig
    from transformers import SegformerForSemanticSegmentation
except ImportError as err:
    raise RuntimeError("install `transformers` to initialize SegFormer") from err


class Segformer(SegformerForSemanticSegmentation):
    def __init__(self, config: Mapping):
        super().__init__(config)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        predict = super().forward(pixel_values)
        return F.interpolate(
            predict.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def _make_head(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


@COMPONENTS.register(
    name="segformer",
    ctype=ComponentType.MODEL,
    supported_tasks=Task.SEMANTIC_SEGMENTATION,
    provider="huggingface",
    description="Segformer for semantic segmentation",
)
def build_segformer(
    msize: str,
    num_cls: int,
    pretrained: bool,
    id2label: dict[int, str] = None,
    label2id: dict[str, int] = None,
    ignore_index: int = 255,
) -> nn.Module:

    _converter = {size: f"b{i}" for i, size in enumerate(["n", "s", "m", "l", "x"])}
    mname = "nvidia/segformer-{}-finetuned-cityscapes-1024-1024".format(
        _converter[msize]
    )

    if pretrained:
        model = Segformer.from_pretrained(mname)
    else:
        hf_config = AutoConfig.from_pretrained(mname)
        model = Segformer(hf_config)

    decode_head = model.decode_head
    model.decode_head.classifier = _make_head(
        decode_head.config.decoder_hidden_size, num_cls
    )
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.semantic_loss_ignore_index = ignore_index

    return model
