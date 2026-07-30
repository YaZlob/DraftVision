import torch
from typing import Any
from PIL import Image
from collections.abc import Callable, Sequence
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.ops import box_convert
from mlcore.core import COMPONENTS, ComponentType, BOXFORMAT


def apply_with_probability(transform, p: float = 1.0):
    if p == 1.0:
        return transform
    return v2.RandomApply([transform], p=p)


def _geometry_params(prms: dict) -> dict:
    fill = prms.get("fill", {})
    if not isinstance(fill, dict):
        raise ValueError(f"if provided `fill` must be <map>, got {type(fill)}")

    _tvt = {
        "image": tv_tensors.Image,
        "mask": tv_tensors.Mask,
        "video": tv_tensors.Video,
    }
    unknown = set(fill).difference(_tvt)
    if unknown:
        raise ValueError(
            f"`fill` contains unexpectd field: {unknown}, expected: {list(_tvt.keys())}"
        )

    # changes provided names aka <image, mask> to corresponded tv_tensor
    _tv_fill = {}
    for key, value in fill.items():
        tv_type = _tvt.get(key)
        _tv_fill[tv_type] = value

    prms["fill"] = _tv_fill
    return prms


# https://docs.pytorch.org/vision/main/_modules/torchvision/transforms/v2/_geometry.html#Resize
@COMPONENTS.register("resize", ctype=ComponentType.TRANSFORMS, provider="torchvision")
class Resize(v2.Resize):
    def __init__(
        self,
        size: int | Sequence[int],
        interpolation: int = v2.InterpolationMode.BILINEAR,
        max_size: int | None = None,
    ):
        super().__init__(size, interpolation, max_size)

    def transform(self, inpt: Any, params: dict) -> Any:
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            if inpt.numel() > 0 and torch.max(inpt) <= 1.0:
                return inpt

        return super().transform(inpt, params)


@COMPONENTS.register(
    "horizontal_flip", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_horizontal_flip(p: float):
    return v2.RandomHorizontalFlip(p)


@COMPONENTS.register(
    "vertical_flip", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_vertical_flip(p):
    return v2.RandomVerticalFlip(p)


@COMPONENTS.register("rotation", ctype=ComponentType.TRANSFORMS, provider="torchvision")
def build_rotation(**params: Any):
    p = params.pop("p")
    augm_prms = _geometry_params(params)
    return apply_with_probability(v2.RandomRotation(**augm_prms), p)


@COMPONENTS.register("affine", ctype=ComponentType.TRANSFORMS, provider="torchvision")
def build_affine(**params: Any):
    p = params.pop("p")
    augm_prms = _geometry_params(params)
    return apply_with_probability(v2.RandomAffine(**augm_prms), p)


@COMPONENTS.register(
    "resized_crop", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_resized_crop(**params: Any):
    return v2.RandomResizedCrop(**params)


@COMPONENTS.register(
    "color_jitter", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_color_jitter(**params: Any):
    return v2.ColorJitter(**params)


@COMPONENTS.register(
    "grayscale", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_grayscale(**params: Any):
    return v2.RandomGrayscale(**params)


@COMPONENTS.register(
    "gaussian_blur", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_gaussian_blur(**params: Any):
    p = params.pop("p")
    return apply_with_probability(v2.GaussianBlur(**params), p)


@COMPONENTS.register(
    "perspective", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_perspective(**params: Any):
    params = _geometry_params(params)
    return v2.RandomPerspective(**params)


@COMPONENTS.register(
    "photometric_distort", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_photometric_distort(**params: Any):
    p = params.pop("p")
    return apply_with_probability(v2.RandomPhotometricDistort(**params), p)


@COMPONENTS.register(
    "random_erasing", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_random_erasing(**params: Any):
    return v2.RandomErasing(**params)


@COMPONENTS.register(
    "random_iou_crop", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_random_iou_crop(**params: Any):
    p = params.pop("p")
    return apply_with_probability(v2.RandomIoUCrop(**params), p)


@COMPONENTS.register(
    "sanitize_bounding_boxes", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_sanitize_bounding_boxes(**params: Any):
    return v2.SanitizeBoundingBoxes(**params)


@COMPONENTS.register("to_dtype", ctype=ComponentType.TRANSFORMS, provider="torchvision")
def build_to_dtype(**params: Any):

    image_dtype = getattr(torch, params.pop("image_dtype", "float32"))
    mask_dtype = getattr(torch, params.pop("mask_dtype", "int64"))
    dtype = {
        tv_tensors.Image: image_dtype,
        tv_tensors.Mask: mask_dtype,
        "others": None,
    }
    return v2.ToDtype(dtype=dtype, **params)


@COMPONENTS.register(
    "ToImage",
    ctype=ComponentType.TRANSFORMS,
    provider="torcvision",
    description="Convert PIL image into tv_tensor.Image",
)
def build_ToImage():
    return v2.ToImage()


@COMPONENTS.register(
    "ConvertImage", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
class ConvertPilImage(v2.Transform):
    _transformed_types = (Image.Image,)

    def __init__(self, dtype: str = "float32", scale: bool = True):
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        inpt = v2.functional.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = tv_tensors.Image(inpt)

        return inpt


@COMPONENTS.register(
    "ConvertBoxes",
    ctype=ComponentType.TRANSFORMS,
    provider="torcvision",
    description="Bounding box converter",
)
class ConvertBoxes(v2.Transform):
    _transformed_types = (tv_tensors.BoundingBoxes,)

    def __init__(self, out_fmt: BOXFORMAT, absolute: bool = False) -> None:
        super().__init__()
        if out_fmt not in BOXFORMAT.__args__:
            msg = f"Expects one of box formats: [{BOXFORMAT.__args__}], got {out_fmt}"
            raise ValueError(msg)

        self.abs = absolute
        self.out_fmt: BOXFORMAT = out_fmt

    def transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        h, w = inpt.canvas_size
        if inpt.format.value.lower() != self.out_fmt:
            in_box_fmt = inpt.format.name.lower()
            inpt = box_convert(inpt, in_fmt=in_box_fmt, out_fmt=self.out_fmt)

        if inpt.numel() > 0:
            is_normalized = torch.max(inpt) <= 1.0
            # Боксы нормализованы, но нужно возвращать абсолютные координаты
            if is_normalized and self.abs:
                scale = torch.tensor([w, h, w, h], dtype=inpt.dtype, device=inpt.device)
                inpt = inpt * scale
            # Боксы в абсолютных координтах и нужна нормализация
            elif not is_normalized and not self.abs:
                scale = torch.tensor([w, h, w, h], dtype=inpt.dtype, device=inpt.device)
                inpt = inpt / scale

        return tv_tensors.BoundingBoxes(inpt, format=self.out_fmt, canvas_size=(h, w))


@COMPONENTS.register(
    "normalize", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_normalize(**params: Any):
    return v2.Normalize(**params)


def _valid_p(p: float) -> None:
    if not isinstance(p, float) or not 0 <= p <= 1:
        raise ValueError("probability must be in [0, 1]")


@COMPONENTS.register(
    "random_choice", ctype=ComponentType.TRANSFORMS, provider="torchvision"
)
def build_random_choice(
    components: list[Callable],
    p: list[float] | None = None,
) -> Callable:
    if not components:
        raise ValueError("transform 'random_choice' requires non-empty components")

    if p is not None:
        if len(p) != len(components):
            raise ValueError("transform 'random_choice' p must match components length")
        for probability in p:
            _valid_p(probability)

        if sum(p) != 1.0:
            raise ValueError("`random_choice` sum probability must be equal to 1")

    return v2.RandomChoice(components, p=p)


@COMPONENTS.register("compose", ctype=ComponentType.TRANSFORMS, provider="torchvision")
def build_compose(components: list[Callable]) -> Callable:
    """Build a sequence of resolved transforms from config."""
    if not components:
        raise ValueError("transform 'compose' requires non-empty components")
    return v2.Compose(components)
