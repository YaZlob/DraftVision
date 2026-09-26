import torch
import random
import numpy as np
import torchvision.transforms.functional as F


from PIL import Image
from typing import TypedDict
from collections.abc import Mapping
from abc import ABC, abstractmethod
from mlcore.data._dataset import Dataset
from mlcore.data._contracts import DetectionTarget, _prepare_det_target
from mlcore.core import (
    ComponentType,
    COMPONENTS,
    ComponentConfig,
    Task,
    CompatibilityError,
)

__all__ = ["Mosaic", "CutMix"]


class TransformMetaData(TypedDict):
    imgtarget: list[list[Image.Image, dict]]
    layout: dict


class BaseMixTransformation(ABC):
    def __init__(self, dataset: Dataset, target_HW: tuple[int, int], p: float):
        if not hasattr(dataset, "load_item"):
            raise RuntimeError("Input `dataset` must have `load_item` method")
        assert 0.0 <= p <= 1.0, "Wrong probability value!"
        assert len(target_HW) == 2 and all([i % 2 == 0 for i in target_HW])

        self.p = p
        self._dataset = dataset
        self._task = dataset.task

        if self._task not in self.compatible_tasks:
            name, task = self.__class__.__name__, self._task
            msg = f"Transform {name} not compatible with task {task}, expected one of {self.compatible_tasks}"
            raise CompatibilityError(msg)

        self._preproc_transorms = COMPONENTS.build(
            ComponentConfig(
                ctype=ComponentType.TRANSFORMS,
                type="compose",
                components=[
                    ComponentConfig(
                        ctype=ComponentType.TRANSFORMS,
                        type="ConvertBoxes",
                        params={"out_fmt": "xyxy", "absolute": True},
                    ),
                    ComponentConfig(
                        ctype=ComponentType.TRANSFORMS,
                        type="resize",
                        params={"size": target_HW},
                    ),
                ],
            )
        )

        # TODO min area in proportions of target hw
        self._postproc_transform = COMPONENTS.build(
            ComponentConfig(
                ctype=ComponentType.TRANSFORMS,
                type="sanitize_bounding_boxes",
                params={"min_area": 1.0},
            )
        )

        self._dlen = len(dataset)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        compatible_tasks = getattr(cls, "compatible_tasks", None)
        if not compatible_tasks:
            raise AttributeError(f"Class {cls.__name__} must define 'compatible_tasks'")

    def _get_imgtarget(self) -> tuple[Image.Image, DetectionTarget]:
        return self._dataset.load_item(random.randint(0, self._dlen - 1))

    @abstractmethod
    def _generate_layout(self, data: TransformMetaData): ...

    @abstractmethod
    def _prepare_data(self, img: Image.Image, target: Mapping) -> TransformMetaData: ...

    @abstractmethod
    def _apply2img(self, data: TransformMetaData) -> Image.Image: ...

    @abstractmethod
    def _apply2target(self, data: TransformMetaData) -> dict: ...

    def apply_transform(
        self, img: Image.Image, target: Mapping
    ) -> tuple[Image.Image, dict]:

        data = self._prepare_data(img, target)
        self._generate_layout(data)
        img = self._apply2img(data)
        target = self._apply2target(data)

        return img, target

    def __call__(self, img: Image.Image, target: Mapping) -> tuple[Image.Image, dict]:
        if random.uniform(0, 1) > self.p:
            img, target = self.apply_transform(img, target)

        return img, target


# Every class requires `dataset`` as additional param
# TODO extend behaviour for classification. Output target labels should be scaled to image parts.
@COMPONENTS.register(
    name="mosaic",
    ctype=ComponentType.TRANSFORMS,
    description="Mosaic augmentation",
)
class Mosaic(BaseMixTransformation):
    compatible_tasks = (Task.OBJECT_DETECTION, Task.CLASSIFICATION)

    def __init__(
        self, dataset: Dataset, p: float, target_HW: tuple[int, int], fill: int = 0
    ):

        super().__init__(dataset, target_HW, p)
        assert isinstance(fill, int) and 0 <= fill <= 255

        if self._task == Task.CLASSIFICATION:
            raise NotImplementedError()

        self.fill = fill / 255
        self.tH, self.tW = target_HW

        self._grid = 4

    def __make_canvas(self) -> torch.Tensor:
        return torch.full((3, 2 * self.tH, 2 * self.tW), self.fill, dtype=torch.float32)

    def _prepare_data(self, img: Image.Image, target: dict) -> TransformMetaData:
        imgtargets = [[img, target]] + [
            self._get_imgtarget() for _ in range(self._grid - 1)
        ]

        for i, (cimg, ctarget) in enumerate(imgtargets):
            cimg, ctarget = self._preproc_transorms(cimg, ctarget)
            imgtargets[i] = [F.pil_to_tensor(cimg) / 255.0, ctarget]

        return TransformMetaData(imgtarget=imgtargets)

    def _generate_layout(self, data: TransformMetaData):
        cx = self.tW + random.randint(-self.tW // 2, self.tW // 2)
        cy = self.tH + random.randint(-self.tH // 2, self.tH // 2)

        x_min, x_max = max(cx - self.tW, 0), min(self.tW + cx, 2 * self.tW)
        y_min, y_max = max(cy - self.tH, 0), min(self.tH + cy, 2 * self.tH)

        layout = {}

        # top left
        xy_shift = (0, self.tH)
        x1c, y1c, x2c, y2c = x_min, self.tH, self.tW, y_max
        x1i, y1i, x2i, y2i = x1c, 0, self.tW, y2c - y1c
        layout[0] = dict(
            shift=xy_shift, cbox=(x1c, y1c, x2c, y2c), ibox=(x1i, y1i, x2i, y2i)
        )

        # top right
        xy_shift = (self.tW, self.tH)
        x1c, y1c, x2c, y2c = self.tW, self.tH, x_max, y_max
        x1i, y1i, x2i, y2i = 0, 0, x2c - x1c, y2c - y1c
        layout[1] = dict(
            shift=xy_shift, cbox=(x1c, y1c, x2c, y2c), ibox=(x1i, y1i, x2i, y2i)
        )

        # bottom left
        xy_shift = (0, 0)
        x1c, y1c, x2c, y2c = x_min, y_min, self.tW, self.tH
        x1i, y1i, x2i, y2i = x_min, y_min, self.tW, self.tH
        layout[2] = dict(
            shift=xy_shift, cbox=(x1c, y1c, x2c, y2c), ibox=(x1i, y1i, x2i, y2i)
        )

        # bottom right
        xy_shift = (self.tW, 0)
        x1c, y1c, x2c, y2c = self.tW, y_min, x_max, self.tH
        x1i, y1i, x2i, y2i = 0, y_min, x2c - x1c, self.tH
        layout[3] = dict(
            shift=xy_shift, cbox=(x1c, y1c, x2c, y2c), ibox=(x1i, y1i, x2i, y2i)
        )

        data["layout"] = layout

    def _apply2img(self, data: TransformMetaData) -> Image.Image:
        canvas = self.__make_canvas()

        imgtargets = data["imgtarget"]
        layout_prms = data["layout"]
        for i, (img, _) in enumerate(imgtargets):
            prms = layout_prms[i]
            x1c, y1c, x2c, y2c = prms["cbox"]
            x1i, y1i, x2i, y2i = prms["ibox"]

            # print(f"Canvas: x1: {x1c:4d} y1: {y1c:4d} x2: {x2c:4d} y2: {y2c:4d}")
            # print(f"Image:  x1: {x1i:4d} y1: {y1i:4d} x2: {x2i:4d} y2: {y2i:4d}")

            canvas[:, y1c:y2c, x1c:x2c] = img[:, y1i:y2i, x1i:x2i]

        return F.to_pil_image(canvas)

    def _apply2target(self, data: TransformMetaData) -> dict:
        clabels, cboxes = [], []
        imgtargets = data["imgtarget"]
        layout_prms = data["layout"]

        for i, (_, target) in enumerate(imgtargets):
            prms = layout_prms[i]
            clabels.append(target["labels"])
            cboxes.append(self._adjust_boxes(target, prms["ibox"], prms["shift"]))

        target = _prepare_det_target(
            boxes=torch.cat(cboxes),
            labels=torch.cat(clabels),
            image_id=target["image_id"],
            img_wh=[2 * self.tW, 2 * self.tH],
            in_box_fmt="xyxy",
        )
        target = self._postproc_transform(target)
        return target

    def _adjust_boxes(
        self, target: dict, borders: tuple[int, int, int, int], shifts: tuple[int, int]
    ):

        boxes = target["boxes"]
        x1i, y1i, x2i, y2i = borders

        boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], x1i, x2i) + shifts[0]
        boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], y1i, y2i) + shifts[1]
        return boxes


class MixUp: ...


def bbox_ioa(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> np.ndarray:

    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    inter_area = (
        torch.minimum(b1_x2[:, None], b2_x2) - torch.maximum(b1_x1[:, None], b2_x1)
    ).clip(0) * (
        torch.minimum(b1_y2[:, None], b2_y2) - torch.maximum(b1_y1[:, None], b2_y1)
    ).clip(
        0
    )

    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (area + eps)


# TODO extend behaviour for classification. Output target labels should be scaled to image parts.
# TODO object aware cutmix
@COMPONENTS.register(
    name="cutmix",
    ctype=ComponentType.TRANSFORMS,
    description="CutMix augmentation",
)
class CutMix(BaseMixTransformation):
    compatible_tasks = (Task.OBJECT_DETECTION, Task.CLASSIFICATION)

    def __init__(
        self,
        dataset: Dataset,
        p: float,
        target_HW: tuple[int, int],
        alpha: float = 1.0,
        retry: int = 3,
    ):

        super().__init__(dataset, target_HW, p)
        self.alpha = alpha
        self.retry = retry
        self._rng = np.random.default_rng()

        if self._task == Task.CLASSIFICATION:
            raise NotImplementedError()

    def _rand_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Generate random bounding box coordinates for the cut region.

        Args:
            width (int): Width of the image.
            height (int): Height of the image.

        Returns:
            (tuple[int]): (x1, y1, x2, y2) coordinates of the bounding box.
        """
        # Sample mixing ratio from Beta distribution
        lam = self._rng.beta(self.alpha, self.alpha)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # Random center
        cx = np.random.randint(width)
        cy = np.random.randint(height)

        # Bounding box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)

        return x1, y1, x2, y2

    def _prepare_data(self, img, target) -> TransformMetaData:
        return TransformMetaData(imgtarget=self._preproc_transorms(img, target))

    def _generate_layout(self, data: TransformMetaData):
        img, target = data["imgtarget"]
        w, h = img.size
        cut_areas = torch.tensor([self._rand_bbox(w, h) for _ in range(self.retry)])

        ioa1 = bbox_ioa(cut_areas, target["boxes"])
        idxs = torch.where(ioa1.sum(1) <= 0)[0]

        if len(idxs) == 0:
            return

        box = cut_areas[random.choice(idxs)]

        img2, target2 = self._preproc_transorms(*self._get_imgtarget())
        ioa2 = bbox_ioa(box.unsqueeze(0), target2["boxes"]).squeeze(0)
        idx2 = torch.where(ioa2 >= 0.1)[0]

        if len(idx2) == 0:
            return

        data["layout"] = dict(img=img2, target=target2, box=box.tolist(), wh=(w, h))

    def _apply2img(self, data: TransformMetaData) -> Image.Image:

        main_img = data["imgtarget"][0]
        try:
            layout = data["layout"]
        except KeyError:
            return main_img

        img2, box = layout["img"], layout["box"]
        patch = img2.crop(box)
        main_img.paste(patch, (box[0], box[1]))

        return main_img

    def _apply2target(self, data: TransformMetaData) -> dict:

        main_target = data["imgtarget"][1]
        try:
            layout = data["layout"]
        except KeyError:
            return main_target

        (x1, y1, x2, y2), target2 = layout["box"], layout["target"]

        boxes2 = target2["boxes"]
        boxes2[:, 0] = torch.clamp(boxes2[:, 0], min=x1, max=x2)
        boxes2[:, 1] = torch.clamp(boxes2[:, 1], min=y1, max=y2)
        boxes2[:, 2] = torch.clamp(boxes2[:, 2], min=x1, max=x2)
        boxes2[:, 3] = torch.clamp(boxes2[:, 3], min=y1, max=y2)

        main_target = _prepare_det_target(
            boxes=torch.cat([main_target["boxes"], boxes2]),
            labels=torch.cat([main_target["labels"], target2["labels"]]),
            image_id=main_target["image_id"],
            img_wh=layout["wh"],
            in_box_fmt="xyxy",
        )

        main_target = self._postproc_transform(main_target)
        return main_target
