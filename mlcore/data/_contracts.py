import torch
from typing import TypedDict
from torchvision import tv_tensors


class DetectionTarget(TypedDict):
    boxes: tv_tensors.BoundingBoxes
    labels: torch.Tensor
    image_id: torch.Tensor
    orig_size: torch.Tensor


class PoseTarget(TypedDict):
    image_id: torch.Tensor
    orig_size: torch.Tensor
    keypoints: tv_tensors.KeyPoints
    boxes: tv_tensors.BoundingBoxes
    vis_mask: torch.Tensor


def load_yolo_ann(path2ann: str) -> tuple[list[int], list[tuple]]:
    labels, boxes = [], []
    with open(path2ann, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            cls_id, *coords = line.split(" ")
            labels.append(int(cls_id))
            boxes.append(list(map(float, coords)))
    return labels, boxes


def _value2tensor(
    elem: torch.Tensor | list | tuple | None, shape: tuple[int, ...]
) -> torch.Tensor:
    if elem is None:
        elem = torch.empty(shape)
    elif isinstance(elem, (list, tuple)):
        elem = torch.tensor(elem)
    elif not isinstance(elem, torch.Tensor):
        raise ValueError()
    return elem


# TODO Универсальная функция для N точек, сейчас ожидается 17
# TODO возможно входные параметры будут пустые, сейчас фильтруется.
# TODO добавить label = 0
def _prepare_pose_target(
    *,
    boxes: list[list[int]],
    keypoints: list[list[int]],
    img_wh: tuple[int, int],
    image_id: int,
) -> PoseTarget:

    kps_tensor = torch.Tensor(keypoints).reshape(-1, 17, 3)
    kps_xy = kps_tensor[:, :, :2]
    kps_mask = kps_tensor[:, :, 2].bool()

    boxes = tv_tensors.BoundingBoxes(
        torch.Tensor(boxes),
        format="xyxy",
        canvas_size=img_wh[::-1],
    )
    keypoints = tv_tensors.KeyPoints(kps_xy, canvas_size=img_wh[::-1])

    target = PoseTarget(
        image_id=image_id,
        orig_size=img_wh,
        keypoints=keypoints,
        boxes=boxes,
        vis_mask=kps_mask,
    )
    return target


# https://docs.pytorch.org/vision/main/generated/torchvision.tv_tensors.BoundingBoxes.html
def _prepare_det_target(
    *,
    boxes: list[tuple] | None,
    labels: list[int] | None,
    image_id: int,
    img_wh: tuple[int, int],
    in_box_fmt: str,
) -> DetectionTarget:

    target = DetectionTarget(
        labels=_value2tensor(labels, (0,)).long(),
        # canvas expects h, w as input
        boxes=tv_tensors.BoundingBoxes(
            _value2tensor(boxes, shape=(0, 4)),
            format=in_box_fmt,
            canvas_size=img_wh[::-1],
        ),
        image_id=(
            image_id
            if isinstance(image_id, torch.Tensor)
            else torch.tensor(image_id, dtype=torch.int64)
        ),
        orig_size=torch.tensor(img_wh, dtype=torch.int64),
    )
    return target
