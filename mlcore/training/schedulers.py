import torch.optim.lr_scheduler as lr_scheduler
from mlcore.core import COMPONENTS, ComponentType
from transformers import get_scheduler


@COMPONENTS.register(
    "one_cycle",
    ctype=ComponentType.SCHEDULER,
    provider="torch",
    description="https://docs.pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.OneCycleLR.html",
)
def build_one_cycle(
    optimizer,
    total_steps: int,
    num_warmup_steps: int,
):
    assert num_warmup_steps < total_steps

    max_lr = [group["lr"] for group in optimizer.param_groups]
    optim_factory = getattr(lr_scheduler, "OneCycleLR")
    return optim_factory(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=num_warmup_steps / total_steps,
    )


# Note! num_warmup_steps - количество батчей для прогрева.
# Если поставить большой Bs & num_warmup_steps, потратишь компьют в пустую
@COMPONENTS.register(
    "constant_with_warmup", ctype=ComponentType.SCHEDULER, provider="hf"
)
def constant_with_warmup(optimizer, total_steps: int, num_warmup_steps: int):
    return get_scheduler(
        "constant_with_warmup", optimizer, num_warmup_steps, total_steps
    )


@COMPONENTS.register("cosine", ctype=ComponentType.SCHEDULER, provider="hf")
def cosine_with_warmup(
    optimizer, total_steps: int, num_warmup_steps: int, num_cycles: float = 0.5
):
    return get_scheduler("cosine", optimizer, num_warmup_steps, total_steps, num_cycles)
