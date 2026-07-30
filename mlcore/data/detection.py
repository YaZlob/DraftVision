import json
import warnings
from PIL import Image

from pathlib import Path
from typing import TypedDict
from collections import defaultdict
from collections.abc import Callable
from mlcore.core import COMPONENTS, ComponentType, Task, BOXFORMAT, IMGEXT

from ._coco import coco2targetcls
from ._dataset import Dataset
from ._is_valid import validate_coco_ann
from ._contracts import DetectionTarget, _prepare_det_target


class DetectionSample(TypedDict):
    image_id: int
    path: Path
    labels: list[int]
    bbox: list[tuple[int, int, int, int]]
    iscrowd: list[bool]


@COMPONENTS.register(
    name="coco_object_detection_dataset",
    ctype=ComponentType.DATASET,
    supported_tasks=Task.OBJECT_DETECTION,
    description="COCO-style object detection dataset",
)
class COCOStyleDataset(Dataset):
    def __init__(
        self,
        img_dir: str | Path,
        annotation: str | Path,
        transforms: Callable | None = None,
        map_coco: bool = False,
    ) -> None:

        super().__init__(transforms=transforms)

        # аннотация COCO2017 начинается с 1 + имеет пропуски, это
        # неудобно для обучения модели. Надо закодировать cat_id
        self._map_coco = map_coco
        self.img_dir = Path(img_dir)
        self.annotation = Path(annotation)
        self.in_box_fmt: BOXFORMAT = "xywh"

        if not self.img_dir.is_dir():
            raise ValueError(f"`img_dir` must be a real directory: {self.img_dir}")
        if not (self.annotation.is_file() and self.annotation.suffix == ".json"):
            raise ValueError(f"`annotation` must be a real file: {self.annotation}")

        self.samples: list[DetectionSample] = self._parse_coco_ann()

    def __len__(self) -> int:
        return len(self.samples)

    def load_item(self, index: int) -> tuple[Image.Image, DetectionTarget]:
        sample: DetectionSample = self.samples[index]
        with Image.open(sample["path"]) as image:
            image = image.convert("RGB")

        target = _prepare_det_target(
            boxes=sample.get("bbox", None),
            labels=sample.get("labels", None),
            image_id=sample.get("image_id"),
            img_wh=image.size,
            in_box_fmt=self.in_box_fmt,
        )
        return image, target

    def _parse_coco_ann(self) -> list[DetectionSample]:
        with self.annotation.open("r", encoding="utf-8") as file:
            data = json.load(file)
        validate_coco_ann(data)

        annotations_by_image = defaultdict(
            lambda: {"labels": [], "boxes": [], "iscrowd": [], "area": []}
        )
        for ann in data.get("annotations"):
            img_data = annotations_by_image[ann["image_id"]]

            img_data["boxes"].append(ann["bbox"])
            cat_id = ann["category_id"]
            if self._map_coco:
                cat_id = coco2targetcls[cat_id]

            img_data["labels"].append(cat_id)
            img_data["iscrowd"].append(ann.get("iscrowd", 0))
            img_data["area"].append(ann["area"])

        samples: list[DetectionSample] = []
        ann_wo_imgs = []

        for image_info in data["images"]:
            img_id = image_info["id"]
            filename = image_info["file_name"]
            path = self._resolve_image_path(filename)
            if not path.exists():
                ann_wo_imgs.append(filename)
                continue

            # if img doesn't contain any annotation then
            # returns emtpy list for every field.
            img_ann = annotations_by_image[img_id]
            if not img_ann.get("boxes"):
                sample = DetectionSample(image_id=img_id, path=path)
            else:
                sample = DetectionSample(
                    image_id=img_id,
                    path=path,
                    bbox=img_ann.get("boxes"),
                    labels=img_ann.get("labels"),
                    iscrowd=img_ann.get("iscrowd"),
                )
            samples.append(sample)

        if ann_wo_imgs:
            msg = f"Not found {len(ann_wo_imgs)} images in `img_dir`!\n"
            msg += ", ".join(ann_wo_imgs[:5])
            raise ValueError(msg)

        return samples

    def _resolve_image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        return self.img_dir / path


# TODO продумать загрузку для разных типов изображений. Сейчас ожидается один тип (img_ext)
@COMPONENTS.register(
    name="yolo_object_detection_dataset",
    ctype=ComponentType.DATASET,
    supported_tasks=Task.OBJECT_DETECTION,
    description="YOLO-style object detection dataset",
)
class YoloStyleDataset(Dataset):
    def __init__(
        self,
        img_dir: str | Path,
        ann_dir: str | Path,
        transforms: Callable | None = None,
        in_box_fmt: BOXFORMAT = "cxcywh",
        img_ext: str = ".jpg",
    ):

        warnings.warn("Only for training! Build coco style annotation for validation!")
        if not in_box_fmt in BOXFORMAT.__args__:
            raise ValueError(
                f"Unsupported input bounding box format: {in_box_fmt}, supports {BOXFORMAT}"
            )

        assert img_ext in IMGEXT

        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.img_ext = img_ext
        self.in_box_fmt = in_box_fmt

        if not self.img_dir.is_dir():
            raise ValueError(f"`img_dir` must be a real directory: {self.img_dir}")
        if not self.ann_dir.is_dir():
            raise ValueError(f"`ann_dir` must be a real directory: {self.ann_dir}")

        self.transforms = transforms
        self.index: list[tuple[int, Path, Path]] = self._make_index()

    @staticmethod
    def _load_ann(path2ann: str) -> tuple[list[int], list[tuple]]:
        labels, boxes = [], []
        with open(path2ann, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                cls_id, *coords = line.split(" ")
                labels.append(int(cls_id))
                boxes.append(list(map(float, coords)))
        return labels, boxes

    # Изображение есть, аннотации нет -> ошибка или нет целевых классов?
    # Пока сделано через аннотацию. Если для изображения аннотации нет, то
    # изображение пропускается.
    def _make_index(self) -> list[tuple[Path, Path]]:
        annotations, ann_wo_imgs = [], []
        for txt in self.ann_dir.iterdir():
            img_name = txt.stem + self.img_ext
            image_path: Path = self.img_dir / img_name
            if image_path.exists():
                annotations.append((image_path, txt))
            else:
                ann_wo_imgs.append(img_name)

        if ann_wo_imgs:
            msg = f"Not found {len(ann_wo_imgs)} images in `img_dir`!\n"
            msg += ", ".join(ann_wo_imgs[:5])
            raise ValueError(msg)
        return annotations

    def __len__(self):
        return len(self.index)

    def load_item(self, index: int) -> tuple[Image.Image, DetectionTarget]:
        img, ann = self.index[index]
        with Image.open(img) as image:
            image = image.convert("RGB")

        labels, boxes = self._load_ann(ann)
        # Для совместисмости с COCO, `image_id` в действительности не используется
        target = _prepare_det_target(
            boxes=boxes,
            labels=labels,
            image_id=index,
            img_wh=image.size,
            in_box_fmt=self.in_box_fmt,
        )
        return image, target
