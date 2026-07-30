import torch
import logging
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from ._solver import BaseSolver, metrics
from mlcore.data._coco import targetcls2coco
from mlcore.utils.visuzlize import save_samples

logger = logging.getLogger()


class DetectionSolver(BaseSolver):
    def __init__(self, config, outdir=None, tuning=None):
        super().__init__(config, outdir, tuning)
        self.visualize = self.runtime.get("visualize", False)
        # На трейне COCO классы мапятся в непрерывный вектор [0, 80]
        # На валидации выходы от [0, 80] мапятся обратно в cat_id COCO
        self.remap_coco = self.runtime.get("map_coco", False)

    def _train_epoch(self) -> float:
        self._model.train()
        if hasattr(self._criterion, "train"):
            self._criterion.train().to(self.device)

        train_loss = grnorm = 0.0
        for i, (samples, targets) in enumerate(self._train_dataloader):
            global_step = self._last_epoch * len(self._train_dataloader) + i
            if global_step == 0 and self.visualize:
                save_samples(
                    samples,
                    targets,
                    output_dir=self.exp_dir,
                    split="train",
                    normalized=True,
                    box_fmt="cxcywh",
                )

            self._optim.zero_grad()
            samples = samples.to(self.device)
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            outputs = self._model(samples, targets=targets)
            loss_dict: dict = self._criterion(outputs, targets)
            loss: torch.Tensor = sum(loss_dict.values())

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("ERROR: Total loss is NaN/Inf before backward!")

            loss.backward()

            if self.grad_clip:
                grnorm += clip_grad_norm_(
                    self._model.parameters(), max_norm=self.grad_clip
                )

            self._optim.step()

            if self._ema:
                self._ema.update(self._model)

            train_loss += loss.item()
            if self.log_time(global_step):
                msg = (
                    f"Train step: {global_step}, average loss: {train_loss / (i + 1):.4f}, "
                    f"average norm: {grnorm / (i+1):.4f} "
                    + ", ".join(
                        [
                            f"prg_{i} = {gr['lr']}"
                            for i, gr in enumerate(self._optim.param_groups)
                        ]
                    )
                )
                logger.info(msg)
                msg = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                logger.info(msg)

                self._writer.add_scalar("Loss-Iter", train_loss / (i + 1), global_step)
                for i, gr in enumerate(self._optim.param_groups):
                    self._writer.add_scalar(f"Optimizer/prg_{i}", gr["lr"], global_step)
                for k, v in loss_dict.items():
                    self._writer.add_scalar(f"Loss/{k}", v.item(), global_step)

            if self._scheduler:
                self._scheduler.step()

        return train_loss / (i + 1)

    @torch.no_grad()
    def _val_epoch(self) -> metrics.ValMetricsOutput:

        model = self._ema if self._ema else self._model
        model.eval()
        if hasattr(self._criterion, "eval"):
            self._criterion.eval()
        self._metrics_tracker.reset()

        for i, (samples, targets) in enumerate(self._val_dataloader):
            global_step = self._last_epoch * len(self._val_dataloader) + i
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]
            samples = samples.to(self.device)
            # orig_size must be in [w, h] because postprocessor scale outputs [w, h, w, h]
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(
                self.device
            )

            # Predicts in xyxy format scaled to `original resolution` from annotation
            # its allows keep boxes unchanged for coco evaluation
            # but boxes must be scaled for `Validator` (targets scaled in metrics)
            # Predicts: list[dict] with keys: labels, boxes, score
            # Targets: list[dict] with keys in coco style from annotation.
            # !Note! Targets scaled to `target resolution` from augmentation
            predicts = model(samples, orig_sizes=orig_sizes)
            # TODO Сохранять предикты, а не таргеты?
            if global_step == 0 and self.visualize:
                save_samples(
                    samples,
                    targets,
                    output_dir=self.exp_dir,
                    split="val",
                    normalized=False,
                    box_fmt="xyxy",
                )

            if self.remap_coco:
                for predict in predicts:
                    predict["labels"] = (
                        torch.tensor(
                            [
                                targetcls2coco[int(x.item())]
                                for x in predict["labels"].flatten()
                            ]
                        )
                        .to(predict["labels"].device)
                        .reshape(predict["labels"].shape)
                    )

            self._metrics_tracker.update(predicts, targets)

        self._metrics_tracker.compute()
        out = metrics.ValMetricsOutput(
            loss=None,
            average_metrics=self._metrics_tracker.average(),
            per_cls_metrics=self._metrics_tracker.per_cls_metrics(),
        )
        return out


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    from mlcore.core import load_experiment_config
    from mlcore.models import load_tuning_state

    conf = load_experiment_config("./configs/dfine.yml")
    solver = DetectionSolver(conf)
    solver.visualize = False

    solver.val_setup()

    state = torch.load("/pretrained/dfine_s_obj2coco.pth", map_location="cpu")
    solver._model = load_tuning_state(solver._model, state)

    val_metrics = solver._val_epoch()
    print(val_metrics.average_metrics)
