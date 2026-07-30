import mlcore.data
from mlcore.core import (
    load_experiment_config,
    Task,
    ExperimentConfig,
    ComponentType,
    ComponentConfig,
    COMPONENTS,
)
from torch.utils.data import DataLoader

if __name__ == "__main__":
    target_task = Task.SEMANTIC_SEGMENTATION
    config: ExperimentConfig = load_experiment_config("/app/configs/segformer.yml")
    if config.task != target_task:
        raise RuntimeError()

    id2label = config.runtime.get("id2label", None)
    if id2label is None:
        raise RuntimeError(f"{target_task} must have `id2label` param")

    total_pix = 0
    pix_per_cls = {cls_id: 1 for cls_id in range(len(id2label))}
    ignore_index = config.runtime.get("ignore_index")

    dataset_conf = config.require_component(ComponentType.DATASET).variant("train")
    dataset_cc = ComponentConfig(
        ctype=dataset_conf.ctype,
        type=dataset_conf.type,
        params=dataset_conf.params,
        components={},
    )

    dataset = COMPONENTS.build(dataset_cc, target_task, "dataset.train")
    params = config.runtime.get("dataloader", config.runtime.get("train_loader", None))

    collate_config = config.get_component(ComponentType.COLLATE_FN)
    collate_fn = None
    if collate_config is not None:
        collate_fn = COMPONENTS.build(collate_config, task=target_task)
        params["collate_fn"] = collate_fn

    dataloader = DataLoader(dataset, **params)

    total_pix = 0
    for img, mask in dataloader:
        total_pix += img.numel()
        valid_mask = mask != ignore_index
        for cls_id in range(len(pix_per_cls)):
            class_mask = (mask == cls_id) & valid_mask
            pix_per_cls[cls_id] += class_mask.sum().item()

    for id, pix_count in pix_per_cls.items():
        print(
            f"CLS: {id2label.get(id):>12} pixels: {pix_count:10d} weight: {total_pix/pix_count:4.2f}"
        )

    weights = [total_pix / pix_count for pix_count in pix_per_cls.values()]
