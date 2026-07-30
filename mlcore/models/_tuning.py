import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger()


def load_tuning_state(
    model: nn.Module, state: dict | str | Path, prob_keys: list[str] = ["model"]
):

    if isinstance(state, (str, Path)):
        state = torch.load(state, map_location="cpu", weights_only=False)

    elif not isinstance(state, dict):
        raise ValueError(f"Expected `state` as [dict | str | Path], got {type(state)}")

    pretrained_weights = None
    for key in prob_keys:
        if key in state:
            pretrained_weights = state[key]
            break
    else:
        raise ValueError(
            f"Provided keys {prob_keys} not found in checkpoint keys: {state.keys()}"
        )

    missed_list, unmatched_list = [], []
    matched_state, model_state = {}, model.state_dict()
    for t, w in model_state.items():
        if t in pretrained_weights:
            if w.shape == pretrained_weights[t].shape:
                matched_state[t] = pretrained_weights[t]
            else:
                unmatched_list.append(t)
        else:
            missed_list.append(t)

    model.load_state_dict(matched_state, strict=False)
    logger.info(f"Missed keys: {missed_list}")
    logger.info(f"Unmatched keys: {unmatched_list}")
    return model
