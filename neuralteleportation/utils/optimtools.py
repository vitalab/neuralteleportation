import torch.nn as nn
from torch.nn import init
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


def initialize_model(model, init_type: str, init_gain: float, non_linearity: str = None) -> nn.Module:
    def init_func(m):
        if init_type == 'none': # use the default initialization
            return
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in',
                                     nonlinearity=non_linearity if non_linearity is not None else "leaky_relu")
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    model.apply(init_func)

    return model