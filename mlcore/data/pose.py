import json
from PIL import Image, ImageDraw
from pathlib import Path
from typing import TypedDict
from ._dataset import Dataset
from collections import defaultdict
from ._contracts import PoseTarget, _prepare_pose_target
from mlcore.utils.coco import COCO_SKELETON, LIMB_COLORS, KEYPOINT_COLORS


class PoseSample(TypedDict):
    image_id: int
    path: Path
    boxes: list[list[int]]
    keypoints: list[list[int]]


class COCOPose(Dataset):
    def __init__(
        self,
        task,
        img_dir: str,
        annotation: str | Path,
        transforms=None,
        mix_transform_config: dict | None = None,
    ):
        super().__init__(task, transforms, mix_transform_config)
        self.img_dir = Path(img_dir)
        self.annotation = Path(annotation)

        if not self.img_dir.is_dir():
            raise ValueError(f"`img_dir` must be a real directory: {self.img_dir}")
        if not (self.annotation.is_file() and self.annotation.suffix == ".json"):
            raise ValueError(f"`annotation` must be a real file: {self.annotation}")

        self.samples: list[PoseSample] = self._parse_coco_ann()

        super().__init__(
            task=task,
            transforms=transforms,
            mix_transform_config=mix_transform_config,
        )

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        return self.img_dir / path

    def _parse_coco_ann(self) -> list[PoseSample]:
        with self.annotation.open("r", encoding="utf-8") as file:
            data = json.load(file)

        annotations_by_image = defaultdict(lambda: {"keypoints": [], "boxes": []})

        # скипнуть неподходящие изобаржения или оставить как негативные примеры?
        # В такой реализации остаются негативные примеры. Чтобы убратьих, поднять if наверх
        for ann in data.get("annotations"):
            image_id = ann["image_id"]
            img_data = annotations_by_image[image_id]

            if ann["num_keypoints"] == 0:
                continue

            img_data["boxes"].append(ann["bbox"])
            img_data["keypoints"].append(ann["keypoints"])

        samples: list[PoseSample] = []
        ann_wo_imgs = []

        for image_info in data.get("images"):
            image_id = image_info["id"]
            filename = image_info["file_name"]
            path = self._resolve_image_path(filename)

            if not path.exists():
                ann_wo_imgs.append(filename)
                continue

            img_ann = annotations_by_image[image_id]
            kps, box = img_ann["keypoints"], img_ann["boxes"]
            # TODO фильтрует изображения без человека. Потом убрать, чтобы были
            # негативные примеры данных
            if not box:
                continue

            samples.append(
                PoseSample(image_id=image_id, path=path, boxes=box, keypoints=kps)
            )

        if ann_wo_imgs:
            msg = f"Not found {len(ann_wo_imgs)} images in `img_dir`!\n"
            msg += ", ".join(ann_wo_imgs[:5])
            raise ValueError(msg)

        return samples

    def load_item(self, index: int) -> tuple[Image.Image, PoseTarget]:
        sample: PoseSample = self.samples[index]
        with Image.open(sample["path"]) as image:
            image = image.convert("RGB")

        target = _prepare_pose_target(
            boxes=sample.get("boxes"),
            keypoints=sample.get("keypoints"),
            img_wh=image.size,
            image_id=sample.get("image_id"),
        )

        return image, target


def draw_coco_pose_pil(
    image,
    kps_xy,
    vis_mask,
    bboxes=None,
    kpt_threshold=0.0,
    line_thickness=3,
    point_radius=5,
):

    draw = ImageDraw.Draw(image)
    N = kps_xy.shape[0]
    print(bboxes.shape)

    # Проходим по каждому человеку (персоне) в батче
    for i in range(N):
        person_kps = kps_xy[i]  # [17, 2]
        person_vis = vis_mask[i]  # [17]
        person_bbox = bboxes[i] if bboxes is not None else None

        # 1. Рисуем баундбокс
        if person_bbox is not None:
            x, y, w, h = [int(v) for v in person_bbox]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=2)

        # 2. Рисуем скелет (линии)
        for start_idx, end_idx in COCO_SKELETON:
            # Рисуем линию только если ОБЕ точки видимы
            if person_vis[start_idx] and person_vis[end_idx]:
                start_pt = (
                    int(person_kps[start_idx, 0]),
                    int(person_kps[start_idx, 1]),
                )
                end_pt = (int(person_kps[end_idx, 0]), int(person_kps[end_idx, 1]))

                color = KEYPOINT_COLORS[start_idx]
                draw.line([start_pt, end_pt], fill=color, width=line_thickness)

        # 3. Рисуем ключевые точки
        for idx in range(17):
            # Рисуем только если точка видима (в бинарной маске это True или 1)
            if person_vis[idx]:
                x, y = person_kps[idx]
                bbox_circle = [
                    x - point_radius,
                    y - point_radius,
                    x + point_radius,
                    y + point_radius,
                ]

                color = KEYPOINT_COLORS[idx]
                # Рисуем заполненный кружок с белой обводкой для контраста
                draw.ellipse(bbox_circle, fill=color, outline=(255, 255, 255), width=1)

    return img


if __name__ == "__main__":
    dataset = COCOPose(
        task="pose_estimation",
        img_dir="/object_detection/coco2017/val2017",
        annotation="/object_detection/coco2017/annotations/person_keypoints_val2017.json",
    )

    img, pose = dataset.load_item(15)
    imgwithskel = draw_coco_pose_pil(
        img, pose.get("keypoints"), pose.get("vis_mask"), bboxes=pose.get("boxes")
    )

    imgwithskel.save("check.webp")
