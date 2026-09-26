from torch.utils.data import DataLoader
from mlcore.core import COMPONENTS, ComponentType, ConfigurationError


@COMPONENTS.register(
    "torch_dataloader",
    ctype=ComponentType.DATALOADER,
    provider="torch",
    description="Default pytorch dataloader factory",
)
def build_torch_dataloader(dataset, **params):
    if params.get("persistent_workers"):
        raise ConfigurationError(
            "persistent_workers=True is incompatible with StopPolicyCallback: "
            "workers survive across epochs and won't observe dataset/collate_fn mutations."
        )
    return DataLoader(dataset, **params)
