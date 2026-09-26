from collections.abc import Mapping
from collections.abc import Callable
from torch.utils.data import Dataset as TorchDataset
from mlcore.core import Task, COMPONENTS, ComponentConfig, ComponentType


class Dataset(TorchDataset):
    """Base map-style dataset with an optional joint image/target transform."""

    def __init__(
        self,
        task: Task,
        transforms: Callable | None = None,
        mix_transform_config: Mapping | None = None,
    ):
        self.task = Task(task)
        self.transforms = transforms
        self._mix_transofrm = self._build_dataset_related_augm(mix_transform_config)

    # TODO правильный проход по transforms с учетом вложенных аугментаций
    def __repr__(self) -> str:
        def _fmt(transform):
            if transform is None:
                return "None"

            try:
                childs = transform.transforms
                return "[" + ", ".join([t.__class__.__name__ for t in childs]) + "]"
            except AttributeError:
                return f"[{transform.__class__.__name__}]"

        return (
            f"{self.__class__.__name__}("
            f"task={self.task.value}, "
            f"transforms={_fmt(self.transforms)}, "
            f"mix_transform={_fmt(self._mix_transofrm)}"
            f")"
        )

    def _build_dataset_related_augm(self, mix_transform_config: Mapping):
        if mix_transform_config is None:
            return None

        config_path = "dataset.mix_transform"
        cconfig = ComponentConfig.from_dict(
            mix_transform_config,
            ctype=ComponentType.TRANSFORMS,
            config_path=config_path,
        )

        return COMPONENTS.build(
            cconfig,
            task=self.task,
            config_path=config_path,
            dataset=self,
        )

    def disable_mix_transform(self):
        self._mix_transofrm = None

    def disable_augm(
        self, compose: ComponentConfig, disable_augm_list: list[str]
    ) -> None:
        if self.transforms is None:
            return

        cconfig = ComponentConfig(
            ctype=ComponentType.TRANSFORMS,
            type="compose",
            components=[
                transform
                for transform in compose.components
                if transform.type not in disable_augm_list
            ],
        )

        self.transforms = COMPONENTS.build(
            config=cconfig,
            task=self.task,
            config_path="Dataset.disable_augm -> components.dataloader.train.dataset.transforms",
        )

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self._mix_transofrm:
            img, target = self._mix_transofrm(img, target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def load_item(self, index):
        raise NotImplementedError(
            "Please implement this function to return item before `transforms`."
        )
