# !Warn! Либа ожидает xyxy боксы на вход для валидации. Конвертация к xywh происходит внутри
# https://github.com/MiXaiLL76/faster_coco_eval/blob/ce61d0e7b18e405442a38f22dddb796a699e312f/faster_coco_eval/utils/pytorch/coco_eval.py#L104
try:
    from faster_coco_eval import COCO
    from faster_coco_eval.utils.pytorch import FasterCocoEvaluator
except ImportError:
    msg = (
        "For standard metrics calculatias such as mAP\n"
        "used https://github.com/MiXaiLL76/faster_coco_eval/tree/main\n"
        "pls install it using `pip install faster-coco-eval`"
    )
    raise ImportError(msg)


from pathlib import Path
from ._base import BaseMetricTracker
from mlcore.core import COMPONENTS, ComponentType, Task
from mlcore.data._coco import targetcls2coco


# TODO добавить сегментацию и ключевые точки, пока только детекция
@COMPONENTS.register(
    "FasterCocoEval",
    ctype=ComponentType.METRIC,
    supported_tasks=[Task.OBJECT_DETECTION],
    provider="https://github.com/MiXaiLL76/faster_coco_eval/tree/main",
    description="modifided version of pycocotools",
)
class CocoEvaluator(BaseMetricTracker):

    metric_names = [
        "AP50:95",
        "AP50",
        "AP75",
        "APsmall",
        "APmedium",
        "APlarge",
        "AR50:95|Dets=1",
        "AR50:95|Dets=10",
        "AR50:95|Dets=100",
        "AR50:95|small",
        "AR50:95|medium",
        "AR50:95|large",
    ]

    def __init__(
        self,
        coco_gt: Path | dict,
        iou_types: list[str] = ["bbox"],
        lvis_style: bool = False,
        ranges={
            "small": [0**2, 32**2],
            "medium": [32**2, 96**2],
            "large": [96**2, 1e5**2],
        },
    ):
        if len(iou_types) != 1 and iou_types[0] == "bbox":
            raise NotImplementedError(
                "Segmentation and Keypoints will be implemented later"
            )

        if isinstance(coco_gt, (str, Path, dict)):
            coco_gt = COCO(coco_gt)
        elif not isinstance(coco_gt, COCO):
            raise ValueError("")

        self._evaluator = FasterCocoEvaluator(coco_gt, iou_types, lvis_style, ranges)

    def __call__(self, predict, gt):
        return self._evaluator.update(predict, gt)

    def update(self, predict: list[dict], gt: list[dict]):
        # {image_id: {labels, boxes, scores}}
        preds = {target["image_id"].item(): pred for pred, target in zip(predict, gt)}
        self._evaluator.update(preds)

    def compute(self):
        self._evaluator.synchronize_between_processes()
        self._evaluator.accumulate()
        self._evaluator.summarize()

    def reset(self):
        self._evaluator.cleanup()

    def average(self) -> dict:
        metrics = self._evaluator.coco_eval["bbox"].stats.tolist()
        result = {}
        for idx, name in enumerate(self.metric_names):
            result[name] = metrics[idx]
        return result

    def per_cls_metrics(self) -> dict:
        return {}
