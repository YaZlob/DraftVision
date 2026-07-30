import re
import torch
import torch.nn as nn
from typing import TypedDict

from mlcore.core import COMPONENTS, ComponentType


# Params required for optimizer
class OptimPrmGroups(TypedDict):
    params: str
    lr: float
    weight_decay: float


# Params required for assigment
class PrmsAssigment:
    pattern: str
    lr: float | None = None
    weight_decay: float | None = None


# based on https://github.com/Peterande/D-FINE/blob/master/src/core/yaml_config.py#L117
def assign_prms_meta(
    model: nn.Module,
    pgassignment: list[PrmsAssigment],
    d_lr: float,
    d_wd: float,
) -> list[dict]:

    pgroups = []
    visited: list[str] = []

    for pga in pgassignment:
        try:
            pattern = re.compile(pga["pattern"])
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern '{pattern}': {e.msg} at position {e.pos}"
            ) from e

        params = {
            k: v
            for k, v in model.named_parameters()
            if v.requires_grad and pattern.match(k)
        }

        pgroups.append(
            OptimPrmGroups(
                params=params.values(),
                lr=pga.get("lr", None) or d_lr,
                weight_decay=pga.get("weight_decay", None) or d_wd,
            )
        )
        visited.extend(list(params.keys()))

    names = [k for k, v in model.named_parameters() if v.requires_grad]
    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {
            k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen
        }
        pgroups.append({"params": params.values()})
        visited.extend(list(params.keys()))

    return pgroups


def _build_torch_optimizer(name: str, parameters, params):
    return getattr(torch.optim, name)(parameters, **params)


@COMPONENTS.register("adam", ctype=ComponentType.OPTIMIZER, provider="torch")
def build_adam(parameters, **params):
    return _build_torch_optimizer("Adam", parameters, params)


@COMPONENTS.register("adamw", ctype=ComponentType.OPTIMIZER, provider="torch")
def build_adamw(parameters, **params):
    return _build_torch_optimizer("AdamW", parameters, params)


@COMPONENTS.register("sgd", ctype=ComponentType.OPTIMIZER, provider="torch")
def build_sgd(parameters, **params):
    return _build_torch_optimizer("SGD", parameters, params)
