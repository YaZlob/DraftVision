import torch
from typing import TypedDict
from torchvision import tv_tensors


class DetectionTarget(TypedDict):
    boxes: tv_tensors.BoundingBoxes
    labels: torch.Tensor
    image_id: torch.Tensor
    orig_size: torch.Tensor


# https://docs.pytorch.org/vision/main/generated/torchvision.tv_tensors.BoundingBoxes.html
def _prepare_det_target(
    *,
    boxes: list[tuple] | None,
    labels: list[int] | None,
    image_id: int,
    img_wh: tuple[int, int],
    in_box_fmt: str,
) -> DetectionTarget:
    labels = torch.tensor(labels) if labels else torch.empty((0,))
    target = DetectionTarget(
        # canvas expects h, w as input
        boxes=tv_tensors.BoundingBoxes(
            torch.tensor(boxes) if boxes else torch.empty((0, 4)),
            format=in_box_fmt,
            canvas_size=img_wh[::-1],
        ),
        labels=labels.long(),
        image_id=torch.tensor(image_id, dtype=torch.int64),
        orig_size=torch.tensor(img_wh, dtype=torch.int64),
    )
    return target
