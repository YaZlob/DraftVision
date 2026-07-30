import torch
from PIL import Image
from torchvision import tv_tensors
from pathlib import Path
from collections.abc import Callable
from mlcore.utils import IMAGE_EXTENSIONS
from mlcore.core import COMPONENTS, ComponentType, Task

from ._dataset import Dataset


@COMPONENTS.register(
    name="semantic_segmentation_dataset",
    ctype=ComponentType.DATASET,
    supported_tasks=Task.SEMANTIC_SEGMENTATION,
    description="dataset for semantic segmentation",
)
class SemanticSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        mask_dir: str | Path,
        transforms: Callable | None = None,
        skip_wo_mask: bool = True,
    ) -> None:

        super().__init__(transforms=transforms)
        root_dir = Path(root_dir)
        mask_dir = Path(mask_dir)
        if not (root_dir.is_dir() and mask_dir.is_dir()):
            raise ValueError(
                "Expected `root_dir` and `mask_dir` to be real directories"
            )
        self.samples: list[tuple[Path, Path]] = self._build_dataset(
            root_dir, mask_dir, skip_wo_mask
        )

    @staticmethod
    def _build_dataset(
        image_dir: Path,
        mask_dir: Path,
        skip_without_mask: bool = True,
    ) -> list[tuple[Path, Path]]:

        samples: list[tuple[Path, Path]] = []
        missing: list[Path] = []

        for image_path in image_dir.iterdir():
            if not image_path.suffix in IMAGE_EXTENSIONS:
                continue

            mask_path = mask_dir / image_path.name
            if mask_path.exists():
                samples.append((image_path, mask_path))
            else:
                missing.append(image_path)

        if not samples:
            raise ValueError("No image/mask pairs found")

        if missing and not skip_without_mask:
            names = ", ".join(path.name for path in missing[:5])
            raise ValueError(f"Missing masks for {len(missing)} images: {names}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def load_item(self, index: int) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = tv_tensors.Image(image)
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask_tensor = tv_tensors.Mask(mask)

        return image_tensor, mask_tensor.to(torch.uint8)


__all__ = ["SemanticSegmentationDataset"]
