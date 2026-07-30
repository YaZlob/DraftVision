import torch
import random
import warnings
import torch.nn.functional as F
from mlcore.core import COMPONENTS, ComponentType, Task
from .._contracts import DetectionTarget


def generate_scales(
    resolution: tuple[int, int],
    base_res_repeat: int,
    scale_factor: float = 0.75,
    upper_scale: bool = True,
) -> list[tuple[int, int]]:
    assert 0.25 <= scale_factor <= 1

    min_res_dim = min(resolution)
    scale_repeat = (min_res_dim - int(min_res_dim * scale_factor / 32) * 32) // 32
    min_scale_calc = lambda dim, iter: int(dim * scale_factor / 32) * 32 + iter * 32

    scales = [
        (min_scale_calc(resolution[0], i), min_scale_calc(resolution[1], i))
        for i in range(scale_repeat)
    ]
    scales += [resolution] * base_res_repeat

    if upper_scale:
        max_scale_calc = (
            lambda dim, iter: int(dim * (2 - scale_factor) / 32) * 32 - iter * 32
        )
        scales += [
            (max_scale_calc(resolution[0], i), max_scale_calc(resolution[1], i))
            for i in range(scale_repeat, 0, -1)
        ]
    return scales


@COMPONENTS.register(
    "BaseDetectorCollateFn",
    ctype=ComponentType.COLLATE_FN,
    supported_tasks=Task.OBJECT_DETECTION,
    description="base collate function for object detection",
)
class BaseImageBatchCollateFunc:
    def __call__(
        self, items: list[tuple[torch.Tensor, DetectionTarget]]
    ) -> tuple[torch.Tensor, list[DetectionTarget]]:
        images: torch.FloatTensor = torch.stack([x[0] for x in items], dim=0)
        targets: list[DetectionTarget] = [x[1] for x in items]
        return images, targets


@COMPONENTS.register(
    "DetectorCollateFn",
    ctype=ComponentType.COLLATE_FN,
    supported_tasks=Task.OBJECT_DETECTION,
    description="Detector collate function with image scale",
)
class ImageBatchCollateFunc(BaseImageBatchCollateFunc):
    def __init__(
        self,
        resolution: tuple[int, int],
        scale_factor: float | None = 0.75,
        use_upper_scale: bool = True,
        base_res_repeat: int | None = 3,
    ):
        warnings.warn(
            "Scales only images, keeps boundig boxes the same!\n"
            "Critical if detector works with unnormalized coodinates"
        )
        self.resolution = resolution
        self.scales = (
            generate_scales(resolution, base_res_repeat, scale_factor, use_upper_scale)
            if scale_factor
            else None
        )

    def __call__(self, items: list[tuple[torch.Tensor, DetectionTarget]]):
        images, targets = super().__call__(items)
        if self.scales:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)

        return images, targets
