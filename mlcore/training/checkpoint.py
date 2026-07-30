from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal

from mlcore.core import ConfigurationError
from mlcore.metrics import ValMetricsOutput

_MONITORMODE = Literal["min", "max"]


@dataclass
class SavingPolicy:
    name: str
    filename: str
    monitor: str
    mode: _MONITORMODE
    best_value: float = -1
    min_delta: float = 0.0

    def __post_init__(self):
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

    def should_save(self, metrics: ValMetricsOutput) -> bool:
        value = _resolve_metric(metrics, self.monitor)
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value

        return value

    @classmethod
    def from_dict(cls, data: Mapping) -> "SavingPolicy":
        return cls(
            name=data.get("name"),
            filename=data.get("filename"),
            monitor=data.get("monitor"),
            mode=data.get("mode"),
            min_delta=data.get("min_delta", 0.0),
        )

    def state_dict(self) -> dict[str:float]:
        return {"best_value": self.best_value}

    def load_state_dict(self, state: dict):
        self.best_value = state["best_value"]


class CheckpointManager:
    def __init__(
        self,
        rules: list[SavingPolicy],
        save_last: bool = True,
        last_filename: str = "last.pth",
    ) -> None:

        self.rules = rules
        self._save_last = save_last
        self._last_filename = last_filename

    @classmethod
    def default(cls) -> "CheckpointManager":
        return cls(
            rules=[
                SavingPolicy(
                    name="best",
                    filename="best.pth",
                    monitor="loss",
                    mode="min",
                )
            ],
            save_last=True,
            last_filename="last.pth",
        )

    @classmethod
    def from_exp_config(cls, policy: dict) -> "CheckpointManager":
        if policy is None:
            return cls.default()

        return cls.from_dict(policy, config_path="config.saving")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config_path: str,
    ) -> "CheckpointManager":
        if not isinstance(data, Mapping):
            raise ConfigurationError(f"{config_path} must be a mapping")

        save_last = data.get("save_last", True)
        last_filename = data.get("last_filename", "last.pth")

        raw_rules = data.get("monitors", None)
        if raw_rules is None:
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
                policy = SavingPolicy.from_dict(rule)
            except Exception as err:
                raise ConfigurationError(
                    f"Failed to create {rule_path}, prms: {rule}"
                ) from err

            rules.append(policy)

        return cls(
            rules=rules,
            save_last=save_last,
            last_filename=last_filename,
        )

    def save(
        self,
        state: dict[str, Any],
        metrics: ValMetricsOutput,
        save_callback: callable,
    ):

        for rule in self.rules:
            should_save = rule.should_save(metrics)
            if not should_save:
                continue

            save_callback(state, rule.filename)

        if self._save_last:
            save_callback(state, self._last_filename)

    def state_dict(self) -> dict:
        return {"rules": [rule.state_dict() for rule in self.rules]}

    def load_state_dict(self, state: dict) -> None:
        rules_state = state.get("rules")
        if len(self.rules) != len(rules_state):
            raise RuntimeError(
                f"Checkpoint Manager expect {len(self.rules)} saving rules, got {len(rules_state)} from state"
            )

        for rule, value in zip(self.rules, rules_state):
            rule.load_state_dict(value)

    def __repr__(self):
        msg = (
            f"Chekpoint Manager: save last = {self._save_last}, "
            f"last filename = {self._last_filename}\nCheckpoint policy:\n"
        )
        for rule in self.rules:
            msg += (
                f"Name: {rule.name}\nfilenameL: {rule.filename}\n"
                f"monitor:{rule.monitor}\nmode:{rule.mode}\nbest_value:{rule.best_value}\n"
            )
        return msg


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
