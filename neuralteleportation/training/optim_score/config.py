from dataclasses import dataclass
from numbers import Number
from typing import Tuple, Callable, Any

from torch import Tensor

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics

ModelEvalFunc = Callable[[NeuralTeleportationModel, Tensor, Tensor, TrainingMetrics, "TeleportationTrainingConfig"],
                         Number]


@dataclass
class TeleportationTrainingConfig(TrainingConfig):
    input_shape: Tuple[int, int, int] = (1, 28, 28)
    teleport_every_n_epochs: int = 2
    num_teleportations: int = 10
    num_batches: int = 1
    cob_range: float = 0.5
    cob_sampling: str = 'usual'
    comparison_metric: Tuple[ModelEvalFunc, Callable[[Any, Any], bool]] = None  # Required
