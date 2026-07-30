import torch


def _totensor(mbtensor: torch.Tensor, device: str):
    if isinstance(mbtensor, (list, tuple)):
        mbtensor = torch.tensor(mbtensor)
    return mbtensor.to(device)


def normalize(
    img: torch.Tensor,
    mean: list | tuple | torch.Tensor,
    std: list | tuple | torch.Tensor,
) -> torch.Tensor:

    assert img.dtype == torch.uint8, "Expected image in unint8 format"

    device = img.device
    std = _totensor(std, device)
    mean = _totensor(mean, device)

    img = img / 255.0

    # Handle both (C, H, W) and (B, C, H, W) tensors
    if img.dim() == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif img.dim() == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {img.dim()}")

    mean, std = mean.to(img.device), std.to(img.device)
    return (img - mean) / std


def revert_normalize(
    img: torch.Tensor,
    mean: list | tuple | torch.Tensor,
    std: list | tuple | torch.Tensor,
) -> torch.Tensor:

    device = img.device
    std = _totensor(std, device)
    mean = _totensor(mean, device)

    # Handle both (C, H, W) and (B, C, H, W) tensors
    if img.dim() == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif img.dim() == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {img.dim()}")

    img = 255 * (img * std + mean)
    uint8_img = img.to(torch.uint8)

    return uint8_img
