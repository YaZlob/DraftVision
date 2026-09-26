import os
from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod
from ._config import ExperimentConfig


class SolverTemplate(ABC):
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

        # Main trainig components
        self._model: Any = None
        self._ema: Any = None
        self._optim: Any = None
        self._scheduler: Any = None
        self._criterion: Any = None
        self._train_dataloader = None
        self._val_dataloader = None

    @abstractmethod
    def _train_epoch(self, *args, **kwargs) -> float:
        """Run full training epoch"""

    @abstractmethod
    def _val_epoch(self, *args, **kwargs) -> dict:
        """Run full validation epoch"""


class CollateFn(ABC):
    @abstractmethod
    def disable(self):
        """Hook for stop policies."""
