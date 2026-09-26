from __future__ import annotations

import warnings
import logging
from collections.abc import Mapping
from typing import Any, Literal
from dataclasses import dataclass, field

from mlcore.core import ConfigurationError
from mlcore.metrics import ValMetricsOutput

logger = logging.getLogger()
_MONITORMODE = Literal["min", "max"]


@dataclass
class MetricPolicy:
    name: str
    metric_name: str
    mode: _MONITORMODE
    min_delta: float = 0.0
    _isimproved: bool = field(init=False)
    _best_value: float = field(init=False)

    def __post_init__(self):
        assert self.mode in _MONITORMODE.__args__
        self._isimproved = False
        self._best_value = float("inf") if self.mode == "min" else float("-inf")

    def update(self, metrics: ValMetricsOutput) -> None:
        value = _resolve_metric(metrics, self.metric_name)
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self._best_value = value

        self._isimproved = improved

    @property
    def best_value(self) -> float:
        return self._best_value

    @best_value.setter
    def best_value(self, value: float):
        self._best_value = value

    @property
    def is_improved(self) -> bool:
        return self._isimproved

    @classmethod
    def from_dict(cls, data: Mapping) -> MetricPolicy:
        return cls(
            name=data.get("name"),
            metric_name=data.get("metric_name"),
            mode=data.get("mode"),
            min_delta=data.get("min_delta", 0.0),
        )

    def state_dict(self) -> float:
        return self.best_value

    def load_state_dict(self, value: float):
        self.best_value = value

    def __repr__(self):
        return f"Monitor: {self.name}, metric: {self.metric_name}, mode: {self.mode}, best_value: {self._best_value}"


def _resolve_metric(metrics: ValMetricsOutput, monitor: str) -> float:
    value: Any = metrics
    for part in monitor.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                raise ConfigurationError(
                    f"Checkpoint monitor {monitor!r} has unknown key {part!r}"
                )
            value = value[part]
            continue

        if not hasattr(value, part):
            raise ConfigurationError(
                f"Checkpoint monitor {monitor!r} has unknown field {part!r}"
            )
        value = getattr(value, part)

    if value is None:
        raise ConfigurationError(f"Checkpoint monitor {monitor!r} resolved to None")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as err:
        raise ConfigurationError(
            f"Checkpoint monitor {monitor!r} must resolve to a number"
        ) from err

    return numeric


class MonitorMetricsTracker:
    def __init__(self, rules: list[MetricPolicy]):
        self.rules: dict[str, MetricPolicy] = {rule.name: rule for rule in rules}

    @classmethod
    def default(cls) -> MonitorMetricsTracker:
        return cls(
            rules=[
                MetricPolicy(
                    name="best",
                    monitor="loss",
                    mode="min",
                )
            ]
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config_path: str,
    ) -> MonitorMetricsTracker:
        if not isinstance(data, Mapping):
            raise ConfigurationError(f"{config_path} must be a mapping")

        try:
            raw_rules = data.get("monitors")
        except KeyError:
            raise ConfigurationError(
                "Provide `monitors` to use saving policy, overwise "
                f"use default policy without `{config_path}` option"
            )

        if not isinstance(raw_rules, list):
            raise ConfigurationError(f"{config_path}.monitors must be a list")

        rules = []
        for index, rule in enumerate(raw_rules):
            rule_path = f"{config_path}.monitors[{index}]"
            if not isinstance(rule, Mapping):
                raise ConfigurationError(f"{rule_path} must be a mapping")

            try:
                policy = MetricPolicy.from_dict(rule)
            except Exception as err:
                raise ConfigurationError(
                    f"Failed to create {rule_path}, prms: {rule}"
                ) from err

            rules.append(policy)
        return cls(rules)

    @classmethod
    def from_exp_config(cls, monitoring: Mapping) -> MonitorMetricsTracker:
        if monitoring is None:
            warnings.warn(
                "\033[35m WARN! `monitoring` not provided, will use default behaviour\033[0m"
            )
            return cls.default()
        return cls.from_dict(monitoring, config_path="config.monitoring")

    def state_dict(self) -> dict[str, dict[str:float]]:
        return {k: v.best_value for k, v in self.rules.items()}

    def load_state_dict(self, rules_state: dict[str:float]) -> None:
        target_k, input_k = self.rules.keys(), rules_state.keys()
        if target_k != input_k:
            missing = target_k - input_k
            extra = input_k - target_k

            msg = f"{self.__class__.__name__} miss rules {missing}"
            if extra:
                msg += f" got extra rules {extra} from state dict"

            raise ValueError(msg)

        for k, v in rules_state.items():
            self.rules[k].load_state_dict(v)

    def __repr__(self):
        msg = f"{self.__class__.__name__}, rules:\n"
        for mp in self.rules.values():
            msg += f"{mp}\n"
        return msg

    def policies(self) -> list[str]:
        return self.rules.keys()

    def update(self, metrics: ValMetricsOutput) -> None:
        for mp in self.rules.values():
            mp.update(metrics)

    def has_improvments(self) -> bool:
        return any([mp.is_improved for mp in self.rules.values()])
