import copy
from dataclasses import dataclass, fields
from typing import Sequence, Callable, Dict, Any

from comet_ml import Experiment
from torch import Tensor
from torch.nn.modules.loss import _Loss

from neuralteleportation.utils.logger import BaseLogger

_SERIALIZATION_EXCLUDED_FIELDS = ['comet_logger', 'exp_logger']


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    device: str = 'cpu'
    cob_range: float = 0.5
    cob_sampling: str = 'within_landscape'
    comet_logger: Experiment = None
    exp_logger: BaseLogger = None
    shuffle_batches: bool = False
    weight_decay: float = 0


@dataclass
class TrainingMetrics:
    criterion: _Loss
    metrics: Sequence[Callable[[Tensor, Tensor], Tensor]]


def config_to_dict(training_config: TrainingConfig) -> Dict[str, Any]:
    """Creates a dict from a ``TrainingConfig`` instance. It replaces the built-in, generic ``asdict`` for dataclasses.

    It is required to customize the conversion of ``TrainingConfig`` objects to dict since complex objects stored in the
    config (i.e. loggers) can't be automatically pickled and cause the built-in ``asdict`` function to crash.
    """
    result = []
    for field in [f for f in fields(training_config) if f.name not in _SERIALIZATION_EXCLUDED_FIELDS]:
        value = copy.deepcopy(getattr(training_config, field.name))
        result.append((field.name, value))
    return dict(result)
