import copy
from dataclasses import dataclass, fields
from typing import Sequence, Callable, Dict, Any, Tuple

from torch import Tensor
from torch.nn.modules.loss import _Loss

from neuralteleportation.utils.logger import BaseLogger


@dataclass
class TrainingConfig:
    optimizer: Tuple[str, Dict[str, Any]] = ("Adam", {"lr": 1e-3})
    lr_scheduler: Tuple[str, str, Dict[str, Any]] = None
    epochs: int = 10
    batch_size: int = 32
    drop_last_batch: bool = False
    device: str = 'cpu'
    logger: BaseLogger = None
    shuffle_batches: bool = False
    max_batch: int = None


@dataclass
class TeleportationTrainingConfig(TrainingConfig):
    cob_range: float = 0.5
    cob_sampling: str = 'intra_landscape'
    every_n_epochs: int = 1
    teleport_only_once: bool = False
    # The ``teleport_fn`` field is required to use the pipeline from the ``training`` module,
    # but it must be declared and initialized by the config classes inheriting from ``TeleportationTrainingConfig``
    # NOTE: Default functions should be set using ``field(default=<function_name>)`` to avoid binding the function
    #       as a method of the dataclass


@dataclass
class TrainingMetrics:
    criterion: _Loss
    metrics: Sequence[Callable[[Tensor, Tensor], float]]


_SERIALIZATION_EXCLUDED_FIELDS = ['logger']


def config_to_dict(training_config: TrainingConfig) -> Dict[str, Any]:
    """Creates a dict from a ``TrainingConfig`` instance. It replaces the built-in, generic ``asdict`` for dataclasses.

    It is required to customize the conversion of ``TrainingConfig`` objects to dict since complex objects stored in the
    config (i.e. loggers) can't be automatically pickled and cause the built-in ``asdict`` function to crash.
    """
    from neuralteleportation.experiments.teleport_training import __training_configs__
    training_config_cls_label = {v: k for k, v in __training_configs__.items()}[training_config.__class__]
    result = {"teleport": training_config_cls_label}
    for field in [f for f in fields(training_config) if f.name not in _SERIALIZATION_EXCLUDED_FIELDS]:
        field_value = getattr(training_config, field.name)
        if callable(field_value):
            field_value = field_value.__name__
        else:
            if type(field_value) is tuple:
                # Tuples cannot be loaded back by the yaml module
                field_value = list(field_value)
            field_value = copy.deepcopy(field_value)
        result[field.name] = field_value
    return result
