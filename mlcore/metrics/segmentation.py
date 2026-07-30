import torch
import evaluate
from collections.abc import Mapping

from mlcore.core import COMPONENTS, ComponentType, Task
from ._base import BaseMetricTracker

COMPONENT_NAME = "seg_metrics"


@COMPONENTS.register(
    name=COMPONENT_NAME,
    ctype=ComponentType.METRIC,
    supported_tasks=Task.SEMANTIC_SEGMENTATION,
    provider="huggingface",
    description="Class for segmentation metrics evaluation",
)
class SegmentationMetricsTracker(BaseMetricTracker):
    def __init__(
        self,
        n_classes: int,
        ignore_index: int,
        id2label: dict[int, str] | None = None,
    ):

        if isinstance(id2label, Mapping):
            if len(id2label) != n_classes:
                raise ValueError(
                    f"{self.__class__.__name__} provide `n_classes` equal to `id2label` num components"
                )
            self.id2label = id2label
        else:
            self.id2label = {i: f"cls_{i}" for i in range(n_classes)}

        self._category_metrics = ["per_category_iou", "per_category_accuracy"]

        self.num_labels = n_classes
        self.ignore_index = ignore_index

        self.reset()

    def reset(self):
        self._metric = evaluate.load("mean_iou")

    def update(self, predict: torch.Tensor, mask: torch.Tensor) -> None:
        self._metric.add_batch(predictions=predict, references=mask)

    def __call__(self, predict, mask):
        self.update(predict, mask)

    def compute(self) -> dict[str, float]:
        self._state = self._metric.compute(
            num_labels=self.num_labels, ignore_index=self.ignore_index
        )

    def average(self) -> dict[str, float]:
        result = {
            "mean_iou": float(self._state["mean_iou"]),
            "mean_acc": float(self._state["mean_accuracy"]),
        }
        return result

    def per_cls_metrics(self) -> dict[str:dict]:
        result = {}
        for group in self._category_metrics:
            cat_list = self._state.get(group).tolist()
            result[group] = {
                label: 100 * cat_list[i] for (i, label) in self.id2label.items()
            }
        return result
