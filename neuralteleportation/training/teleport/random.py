import random
from dataclasses import dataclass, field
from typing import Callable

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TeleportationTrainingConfig


def teleport_model_randomly(model: NeuralTeleportationModel, config: "RandomTeleportationTrainingConfig", **kwargs) \
        -> NeuralTeleportationModel:
    if random.random() < config.teleport_prob:
        print("Applying random COB to model in training")
        model.random_teleport(
            cob_range=config.cob_range, sampling_type=config.cob_sampling)
    else:
        print("Skipping COB")

    return model


@dataclass
class RandomTeleportationTrainingConfig(TeleportationTrainingConfig):
    teleport_fn: Callable = field(default=teleport_model_randomly)
    teleport_prob: float = 1.  # By default, always teleport when reaching `teleport_every_n_epochs`
