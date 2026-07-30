from ._base import BaseMetricTracker
from mlcore.core import COMPONENTS, ComponentType, Task


@COMPONENTS.register(
    "MergedMetrics",
    ctype=ComponentType.METRIC,
    supported_tasks=[
        Task.CLASSIFICATION,
        Task.OBJECT_DETECTION,
        Task.SEMANTIC_SEGMENTATION,
    ],
    description="Merge several metrics into one",
)
class MergedMetrics(BaseMetricTracker):
    def __init__(self, **trackers: BaseMetricTracker):
        if not trackers:
            raise ValueError("MergedMetrics requires at least one metric")

        for name, tracker in trackers.items():
            if not isinstance(tracker, BaseMetricTracker):
                raise ValueError(
                    f"Metric {name} must inherit BaseMetricTracker, got {tracker}"
                )

        self._trackers = trackers

    def reset(self):
        for tracker in self._trackers.values():
            tracker.reset()

    def update(self, predict, gt):
        for tracker in self._trackers.values():
            tracker.update(predict, gt)

    def compute(self):
        for tracker in self._trackers.values():
            tracker.compute()

    def average(self):
        results = {}
        for tracker in self._trackers.values():
            results.update(tracker.average())
        return results

    def per_cls_metrics(self):
        results = {}
        for tracker in self._trackers.values():
            results.update(tracker.per_cls_metrics())
        return results
