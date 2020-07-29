from torch.optim import Optimizer


def get_optimizer_lr(optimizer: Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def update_optimizer_params(optimizer: Optimizer, new_state) -> Optimizer:
    optim_state = optimizer.state_dict()
    if "params" in new_state["param_groups"][0].keys():
        del new_state["param_groups"][0]["params"]
    optim_state["param_groups"][0].update(new_state["param_groups"][0])
    optimizer.load_state_dict(optim_state)
    return optimizer
