from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.nn.functional import normalize

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TeleportationTrainingConfig


def simulate_teleportation_sphere(model: NeuralTeleportationModel, config: "PseudoTeleportationTrainingConfig",
                                   **kwargs) -> NeuralTeleportationModel:
    print(f"Shifting weights on a sphere similar to a {config.cob_sampling} teleportation w/ {config.cob_range} "
          f"COB range.")

    model.cpu()  # Move model to CPU to avoid having 2 models on the GPU (to avoid possible CUDA OOM error)

    teleported_model = deepcopy(model).random_teleport(cob_range=config.cob_range,
                                                       sampling_type=config.cob_sampling)

    init_layers = model.get_weights(concat=False)
    teleported_layers = teleported_model.get_weights(concat=False)

    pseudo_teleported_layers = []
    for init_layer, teleported_layer in zip(init_layers, teleported_layers):
        layer_shift = torch.randn_like(init_layer)
        layer_shift = normalize(layer_shift, p=1, dim=0) * torch.norm(teleported_layer - init_layer, 1)
        pseudo_teleported_layer = init_layer + layer_shift
        pseudo_teleported_layers.append(pseudo_teleported_layer)

    pseudo_teleported_weights = torch.cat(pseudo_teleported_layers)
    model.set_weights(pseudo_teleported_weights)
    return model.to(config.device)


def simulate_teleportation_distribution(model: NeuralTeleportationModel,
                                        config: "DistributionTeleportationTrainingConfig",
                                        **kwargs) -> NeuralTeleportationModel:
    print(f"Shifting weights to a similar distribution to a {config.cob_sampling} teleportation w/ {config.cob_range}"
          f"COB range.")

    model.cpu()
    teleported_model = deepcopy(model).random_teleport(cob_range=config.cob_range,
                                                       sampling_type=config.cob_sampling)
    teleported_weights = teleported_model.get_weights(concat=True).cpu().detach().numpy()
    hist, bin_edges = np.histogram(teleported_weights, bins=1000)
    hist = hist / hist.sum()
    model.init_like_histogram(hist, bin_edges)
    return model.to(config.device)


@dataclass
class DistributionTeleportationTrainingConfig(TeleportationTrainingConfig):
    teleport_fn : Callable = field(default=simulate_teleportation_distribution)


@dataclass
class PseudoTeleportationTrainingConfig(TeleportationTrainingConfig):
    teleport_fn: Callable = field(default=simulate_teleportation_sphere)
