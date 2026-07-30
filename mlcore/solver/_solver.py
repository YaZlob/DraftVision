import os
import logging
import atexit
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod
from functools import cached_property

import mlcore.data
import mlcore.models as models
import mlcore.training as training
import mlcore.criterion
import mlcore.log as _log
import mlcore.metrics as metrics
from mlcore.core import (
    COMPONENTS,
    ComponentType,
    ConfigurationError,
    ExperimentConfig,
    Registry,
    Task,
    _TARGETS,
    save_experiment_config,
    load_experiment_config,
    CONFIGNAME,
    BuildComponentError,
)
from mlcore.training.checkpoint import CheckpointManager

logger = logging.getLogger()


def _validate_provided_dir(folder: str | Path | None) -> Path:
    path_obj = Path(folder)
    if not path_obj.is_dir():
        raise FileNotFoundError(f"The folder '{folder}' does not exist.")

    return path_obj


class BaseSolver(ABC):

    def __init__(
        self,
        config: ExperimentConfig,
        outdir: str | Path | None = None,
        tuning: str | Path | None = None,
    ) -> None:

        self._tuning = None
        if tuning and os.path.isfile(tuning):
            self._tuning = Path(tuning)

        self._outdir: Path = outdir
        self._expdir: Path = None
        self._config: ExperimentConfig = config
        self._registry: Registry = COMPONENTS
        self._writer: SummaryWriter = None

        self._device = "cuda"
        self._last_epoch = 0
        self._model: nn.Module = None
        self._ema: nn.Module = None
        self._optim: nn.Module = None
        self._scheduler: nn.Module = None
        self._criterion: nn.Module = None
        self._metrics_tracker: metrics.BaseMetricTracker = None
        self._train_dataloader = None
        self._val_dataloader = None

        # auxiliary params
        self.task: Task = config.task
        self._print_freq = config.runtime.get("print_freq", 1000)

    def _build_model(self) -> nn.Module:
        model = self._registry.build(
            self._config.require_component(ComponentType.MODEL),
            task=self.task,
        )
        return model

    @property
    def runtime(self) -> dict:
        return self._config.runtime

    @property
    def model_provider(self) -> str:
        model_conf = self._config.require_component(ComponentType.MODEL)
        component = self._registry.resolve_spec(
            model_conf.type, model_conf.ctype, self.task
        )
        return component.provider

    @property
    def outdir(self) -> Path:
        # force outdir from initialization params
        self._outdir = Path(self._outdir or self._config.outdir)
        if not self._outdir.exists():
            raise ValueError(f"Provided `outdir` {str(self._outdir)} not found")
        return self._outdir

    @cached_property
    def run_name(self, max_d: int = 3) -> str:

        count, max_run = 1, (10**max_d)

        model_conf = self._config.require_component(ComponentType.MODEL)
        model = model_conf.type
        size = model_conf.params.get("size", None)

        template = f"{model}_{size}" if size else f"{model}"
        for i in range(count, max_run):
            name = f"{template}_{i:0{max_d}d}"
            if not (self.outdir / name).exists():
                return name

        raise RuntimeError(
            "Failed to generate new run name. Probably exceed `max_d` for experiments."
        )

    @property
    def exp_dir(self) -> Path:
        """Creates exp dir with name like <model>_<msize>_<run_id>"""
        if not isinstance(self._expdir, Path):
            self._expdir = self.outdir / self.run_name
        return self._expdir

    @exp_dir.setter
    def exp_dir(self, value: Path):
        if not isinstance(value, Path) and value.is_dir():
            raise ValueError
        self._expdir = value

    def _loader_params(self, target: str) -> dict[str, Any]:
        runtime_conf = self._config.runtime
        params = runtime_conf.get(
            "dataloader", runtime_conf.get(f"{target}_loader", None)
        )
        if params is None:
            raise ConfigurationError(
                "Not found any params for dataloder. Cpecify directly "
                "depening on target with keys '<train | test>_loader` "
                "or use same params for all targets with `dataloader`"
            )

        params.setdefault("shuffle", target == "train")
        return params

    def _build_dataloader(self, target: str) -> Any:
        if target not in _TARGETS:
            raise ValueError("Wrong target for making dataloader")

        dloader_conf = self._config.require_component(ComponentType.DATALOADER)
        dataloader = self._registry.build(
            dloader_conf.variant(target),
            task=self.task,
            config_path=f"dataloader.{target}",
        )
        return dataloader

    def _build_optimizer(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError(
                "Building optimizer is possible only after building model"
            )

        optim_conf = self._config.require_component(ComponentType.OPTIMIZER)
        optim_prms = optim_conf.params

        # TODO Добавить проверку необходимых параметров обязательных компонентов?
        if "lr" not in optim_prms:
            raise BuildComponentError("`lr` must be specified for optimizer")

        if "weight_decay" not in optim_prms:
            raise BuildComponentError("'weight_decay' must be specified for optimizer")

        pgassignment: list = optim_prms.pop("pgassignment", None)
        if pgassignment:
            if not isinstance(pgassignment, (list, tuple)):
                raise BuildComponentError(
                    f"`pgassignment` must be list, got {type(pgassignment)}"
                )
            pgroups = training._optimizers.assign_prms_meta(
                self._model,
                pgassignment,
                d_lr=optim_prms.get("lr"),
                d_wd=optim_prms.get("weight_decay"),
            )
        else:
            pgroups = self._model.parameters()

        optim = self._registry.build(
            optim_conf, config_path="components.optimizer", parameters=pgroups
        )

        return optim

    def _build_criterion(self) -> nn.Module:
        config = self._config.require_component(ComponentType.LOSS)
        weight = config.params.get("weight", None)

        if weight:
            weight_tensor = torch.tensor(weight, dtype=torch.float32).to(self.device)
            config.params["weight"] = weight_tensor

        criterion = self._registry.build(config, self.task)
        return criterion

    def _build_scheduler(self) -> nn.Module | None:
        if self._optim is None:
            raise RuntimeError(
                "Building scheduler is possible only after building optimizer"
            )

        scheduler_config = self._config.get_component(ComponentType.SCHEDULER)
        if scheduler_config is None:
            return None

        scheduler = self._registry.build(
            scheduler_config,
            optimizer=self._optim,
            config_path="config.scheduler",
            total_steps=self.epochs * len(self._train_dataloader),
        )
        return scheduler

    def _build_ema(self) -> nn.Module:
        ema_config = self._config.get_component(ComponentType.EMA)
        if ema_config is None:
            return None

        ema = self._registry.build(
            ema_config, config="components.ema", model=self._model
        )
        return ema

    def _build_metrics_tracker(self) -> metrics.BaseMetricTracker:
        metrics_config = self._config.get_component(ctype=ComponentType.METRIC)
        if metrics_config is None:
            return None
        metrics_tracker = self._registry.build(config=metrics_config, task=self.task)
        return metrics_tracker

    def _build_checkpoint_manager(self) -> CheckpointManager:
        return CheckpointManager.from_dict(
            self._config.saving, config_path="config.saving"
        )

    def _build_tb_writer(self) -> SummaryWriter:
        tb_log_dir = self.runtime.get("tb_log_dir", None)
        swd = os.path.join(tb_log_dir, self.run_name) if tb_log_dir else self.exp_dir
        writer = SummaryWriter(swd)
        atexit.register(writer.close)
        return writer

    def _state(self) -> dict[str, Any]:
        return {
            "model": self._model.state_dict(),
            "last_epoch": self._last_epoch,
            "optimizer": self._optim.state_dict(),
            "scheduler": self._scheduler.state_dict() if self._scheduler else {},
            "ema": self._ema.state_dict() if self._ema else {},
            "chkpt": self._checkpoint_manager.state_dict(),
        }

    def _load_state(self, state: dict[str:Any]):
        self._last_epoch = state.get("last_epoch")
        self._model.load_state_dict(state["model"])
        self._optim.load_state_dict(state["optimizer"])

        # Optional components
        if self._scheduler and "scheduler" in state:
            self._scheduler.load_state_dict(state["scheduler"])
        if self._ema and "ema" in state:
            self._ema.load_state_dict(state["ema"])
        if "chkpt" in state:
            self._checkpoint_manager.load_state_dict(state["chkpt"])

    def _save(self, state: dict, checkpoint: str):
        torch.save(state, self._expdir / checkpoint)

    @property
    def device(self) -> str:
        return self._device

    def log_time(self, global_step: int) -> bool:
        return global_step % self._print_freq == 0

    @device.setter
    def device(self, value: str):
        if value not in ["cpu", "cuda"]:
            raise ValueError(f"Unexpected device: {value} expected ['cuda', 'cpu']")
        self._device = value

    # TODO several gpus
    def train_setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Training works only with cuda available")

        if self.device != "cuda":
            raise ValueError("Use `cuda` for training")

        # Parse runtime parameters
        self.epochs = self.runtime.get("epochs")
        self.grad_clip = self.runtime.get("grad_clip", 1.0)

        # Create modules using registry
        self._model = self._build_model().to(self.device)
        if self._tuning:
            logger.info(f"Load tuning checkpoint from {self._tuning}")
            self._model = models.load_tuning_state(self._model, self._tuning)

        self._criterion = self._build_criterion()

        self._train_dataloader = self._build_dataloader("train")
        self._val_dataloader = self._build_dataloader("val")

        self._ema = self._build_ema()
        self._optim = self._build_optimizer()
        self._scheduler = self._build_scheduler()
        self._metrics_tracker = self._build_metrics_tracker()
        self._checkpoint_manager = self._build_checkpoint_manager()

    def val_setup(self) -> None:
        _log.pylog_config()
        self._model = self._build_model().to(self.device)
        self._criterion = self._build_criterion()
        self._val_dataloader = self._build_dataloader("val")
        self._metrics_tracker = self._build_metrics_tracker()
        self._checkpoint_manager = self._build_checkpoint_manager()

    def _training_loop(self) -> dict[str, float]:
        """Run the full training loop"""

        if self._model is None:
            raise RuntimeError("Call `train_setup` before start fit model")

        for epoch in range(self._last_epoch, self.epochs):
            train_loss = self._train_epoch()
            val_metrics: metrics.ValMetricsOutput = self._val_epoch()
            self._last_epoch += 1

            msg = (
                f"Model: {self._model.__class__.__name__} Epoch: [{epoch + 1:3d}/{self.epochs}], "
                f"Train Loss: {train_loss:5.3f}"
            )
            self._writer.add_scalar("Loss-Epoch/Train", train_loss, epoch)
            if val_metrics.loss:
                msg += f" | Validation Loss: {val_metrics.loss:5.3f}"
                self._writer.add_scalar("Loss-Epoch/Val", val_metrics.loss, epoch)
            logger.info(msg)

            if val_metrics.average_metrics:
                logger.info(
                    " | ".join(
                        [
                            f"{key}={value:.3f}"
                            for key, value in val_metrics.average_metrics.items()
                        ]
                    )
                )
                for key, value in val_metrics.average_metrics.items():
                    self._writer.add_scalar(f"Validation/{key}", value, epoch)

            self._checkpoint_manager.save(
                state=self._state(), metrics=val_metrics, save_callback=self._save
            )

    @abstractmethod
    def _train_epoch(self, *args, **kwargs) -> float:
        """Run full training epoch"""

    @abstractmethod
    def _val_epoch(self, *args, **kwargs) -> metrics.ValMetricsOutput:
        """Run full validation epoch"""

    def fit(self):
        # Prepare experiment directory
        self.exp_dir.mkdir()
        self._config.outdir = str(self.exp_dir)

        _log.pylog_config(self.exp_dir)
        self._writer = self._build_tb_writer()
        self._writer.add_text("Train config", str(self._config), 0)
        save_experiment_config(self._config, self.exp_dir / CONFIGNAME)

        self.train_setup()
        self._training_loop()

    @classmethod
    def resume(cls, folder: str | Path):
        """Resume training using last checkpoint in experiment folder"""

        folder: Path = _validate_provided_dir(folder)
        exp_conf: ExperimentConfig = load_experiment_config(folder / CONFIGNAME)
        _log.pylog_config(folder)

        logging.info(f"Resuming training from {folder} ... load state")

        checkpoint: Path = folder / "last.pth"
        if not checkpoint.is_file():
            raise ValueError(f"`last.pth` not found in provided dir: {folder}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)

        solver: BaseSolver = cls(exp_conf)
        solver.train_setup()
        solver._load_state(state)
        solver.exp_dir = folder
        solver._writer = solver._build_tb_writer()
        solver._training_loop()

    @classmethod
    def evalualte(cls, folder: str) -> None:
        """Evaluates already trained model"""

        folder: Path = _validate_provided_dir(folder)
        exp_conf = load_experiment_config(folder / CONFIGNAME)
        solver: BaseSolver = cls(exp_conf)
        solver.val_setup()
        solver.exp_dir = folder

        logger.info(
            f"Start evaluation for {solver._config.name}, provided checkpoint dir: {folder}"
        )
        for rule in solver._checkpoint_manager.rules:
            chkpt = rule.filename
            try:
                state = torch.load(
                    folder / chkpt, map_location="cpu", weights_only=False
                )
                solver._model.load_state_dict(state["model"], strict=True)
            except Exception as err:
                logger.error(f"Load state for {chkpt} err: {err}")
                continue

            logger.info(
                f"Checkpoint {chkpt} monitoring {rule.monitor}. Epoch: {state.get('last_epoch')}"
            )

            val_metrics = solver._val_epoch()
            logger.info(val_metrics)
