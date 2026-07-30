from torch.utils.data import DataLoader
from mlcore.core import COMPONENTS, ComponentType


@COMPONENTS.register(
    "torch_dataloader",
    ctype=ComponentType.DATALOADER,
    provider="torch",
    description="Default pytorch dataloader factory",
)
def build_torch_dataloader(dataset, **params):
    return DataLoader(dataset, **params)
