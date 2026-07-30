from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class BaseMetricTracker(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear internal state buffer"""

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    @abstractmethod
    def compute(self):
        """Computes metrics based on accumulated info"""

    @abstractmethod
    def update(self, predict: Any, gt: Any) -> None:
        """Updates internal state buffer"""

    @abstractmethod
    def average(self) -> dict[str, float]:
        """Calculates average metrics
        returns mapping in format: metric:value"""

    @abstractmethod
    def per_cls_metrics(self) -> dict[str : dict[str:float]]:
        """Calculates per class metrics
        returns mapping in format:
        metric: { cls_id (str): value (float) }
        """


@dataclass(frozen=True, slots=True)
class ValMetricsOutput:
    loss: float | None
    average_metrics: dict[str, float] = field(default_factory=dict)
    per_cls_metrics: dict[str, dict[str | int : float]] | None = field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.loss is None and self.average_metrics is None:
            raise ValueError("At least `loss` or `average_metrics` must be set")

    def __repr__(self) -> str:
        msg = f"Validation metrics: loss {self.loss or 0.0 :.4f}\n"
        if self.average_metrics:
            msg += (
                " | ".join(
                    [
                        f"{key}={value:.3f}"
                        for key, value in self.average_metrics.items()
                    ]
                )
                + "\n"
            )

        if self.per_cls_metrics:
            metrics = list(self.per_cls_metrics.keys())
            classes = sorted(
                {c_id for metric in self.per_cls_metrics.values() for c_id in metric}
            )

            header = ["cls_id"] + metrics
            widths = [len(h) for h in header]

            rows = []
            for c_id in classes:
                row = [str(c_id)] + [
                    f"{self.per_cls_metrics[metric].get(c_id, 0.0):.3f}"
                    for metric in metrics
                ]
                rows.append(row)

            for row in rows:
                # skip cls_id
                for i in range(1, len(row)):
                    widths[i] = max(widths[i], len(row[i]))

            msg += (
                " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(header))
                + "\n"
            )

            msg += "-+-".join("-" * w for w in widths) + "\n"
            for row in rows:
                msg += (
                    " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
                    + "\n"
                )

        return msg
