import torch
import logging

from typing import TypedDict
from pathlib import Path
from mlcore.data._dataset import Dataset
from mlcore.core import CallBackRegistry, SolverTemplate, Callback, CollateFn
from .monitoring import MonitorMetricsTracker, MetricPolicy

logger = logging.getLogger()
CALLBACKS = CallBackRegistry()


@CALLBACKS.register(name="stop_policy", when="on_epoch_start", persistent=False)
class StopPolicyCallback(Callback):
    def __init__(
        self,
        solver: SolverTemplate,
        trigger_epoch: int,
        disable_augm_list: list[str] = None,
        disable_mix_augm: bool = False,
    ):

        super().__init__(solver)
        self.trigger_epoch = trigger_epoch
        self.disable_augm_list = disable_augm_list
        self.disable_mix_augm = disable_mix_augm

    def on_epoch_start(self):
        if self._solver.epoch >= self.trigger_epoch and not self.is_triggered:
            self._stop_policy()
            self.is_triggered = True

    def _stop_policy(self) -> None:
        dataloader = self._solver._train_dataloader

        collate_fn = dataloader.collate_fn
        if isinstance(collate_fn, CollateFn):
            collate_fn.disable()

        dataset: Dataset = dataloader.dataset

        logger.info(f"Dataset: {dataset}")
        if self.disable_mix_augm:
            dataset.disable_mix_transform()

        # Compose (main transform augmentation)
        try:
            compose = (
                self._solver._config.components["dataloader"]
                .variant("train")
                .components["dataset"]
                .components["transforms"]
            )
            dataset.disable_augm(compose, self.disable_augm_list)
        except Exception as err:
            err.add_note(
                f"{self.__class__.__name__} failed change transform list\n"
                f"Disable augms: {self.disable_augm_list}, config_path: `dataloader.train.dataset.transforms`"
            )
            raise
        logger.info(f"Dataset after stop policy: {dataset}")


class EarlyStopError(Exception):
    """Raised when the number of epochs without improvement exceeds patience."""


@CALLBACKS.register(name="early_stop", when="on_epoch_end", persistent=True)
class EarlyStopCallback(Callback):
    def __init__(self, solver: SolverTemplate, patience: int):
        super().__init__(solver=solver)

        assert isinstance(patience, int) and patience >= 1

        self.patience = patience
        self._wait = 0

    def on_epoch_end(self):
        logger.debug(
            f"call {self.__class__.__name__}, epoch: {self._solver._last_epoch}"
        )
        monitor: MonitorMetricsTracker = self._solver._monitoring

        if monitor.has_improvments():
            self._wait = 0
            return

        self._wait += 1
        if self._wait > self.patience:
            raise EarlyStopError()


class Target(TypedDict):
    monitor: str
    filename: str


@CALLBACKS.register(name="saving", when="on_epoch_end", persistent=True)
class SaveCallback(Callback):
    def __init__(self, solver: SolverTemplate, targets: list[Target]):
        assert isinstance(targets, list)
        assert all([isinstance(item, dict) for item in targets])
        super().__init__(solver=solver)

        subscription = set([target["monitor"] for target in targets])
        monitors = set(self._solver._monitoring.policies())

        if not subscription.issubset(monitors):
            msg = "Monitors targets must be a subset of monitoring polices\n"
            msg += f"Targets: {subscription}\nMonitoring:{monitors}"

            raise ValueError(msg)

        self.targets = targets

    def on_epoch_end(self):
        logger.debug(
            f"call {self.__class__.__name__}, epoch: {self._solver._last_epoch}"
        )
        monitor: MonitorMetricsTracker = self._solver._monitoring
        state = self._solver._state()
        for target in self.targets:
            metric = target["monitor"]
            policy: MetricPolicy = monitor.rules.get(metric)
            if policy.is_improved:
                logger.info(
                    f"Metric: {policy.metric_name} was improved, new value: {policy.best_value:.4f}"
                )
                self._save(state, self._solver.exp_dir / target.get("filename"))

        self._save(state, self._solver.exp_dir / "last.pth")

    def _save(self, state: dict, checkpoint: Path):
        torch.save(state, checkpoint)
