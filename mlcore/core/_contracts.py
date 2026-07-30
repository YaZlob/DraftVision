from enum import StrEnum
from typing import Literal
from collections.abc import Sequence
from ._errors import ConfigurationError

BOXFORMAT = Literal["xyxy", "xywh", "cxcywh"]
IMGEXT = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class Task(StrEnum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class ComponentType(StrEnum):
    MODEL = "model"
    LOSS = "loss"
    METRIC = "metric"
    DATASET = "dataset"
    DATALOADER = "dataloader"
    TRANSFORMS = "transforms"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    COLLATE_FN = "collate_fn"
    EMA = "ema"


REQUIRED_COMPONENT_TYPES = frozenset(
    {
        ComponentType.MODEL,
        ComponentType.DATASET,
        ComponentType.DATALOADER,
        ComponentType.OPTIMIZER,
        ComponentType.TRANSFORMS,
        ComponentType.LOSS,
    }
)


def _get_ctype(
    value: str | ComponentType, config_path: str | None = None
) -> ComponentType:
    if isinstance(value, ComponentType):
        return value
    try:
        return ComponentType(value)
    except Exception as err:
        msg = (
            f"provided item must be one of "
            + ", ".join(ctype.value for ctype in ComponentType)
            + f" got `{value}`"
        )
        if config_path and isinstance(config_path, str):
            msg += f"\nconfig_path: {config_path}"

        raise ConfigurationError(msg) from err


def _get_task(value: object) -> frozenset[Task] | None:
    if value is None:
        return None
    if isinstance(value, Task):
        return frozenset([value])
    if isinstance(value, str):
        return frozenset([Task(value)])
    if isinstance(value, Sequence):
        return frozenset([Task(item) for item in value])

    raise ValueError("task must be a string or sequence of strings")
