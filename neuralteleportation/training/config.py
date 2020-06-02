from dataclasses import dataclass
from typing import Sequence, Callable

from torch import Tensor
from torch.nn.modules.loss import _Loss


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    device: str = 'cpu'


@dataclass
class TrainingMetrics:
    criterion: _Loss
    metrics: Sequence[Callable[[Tensor, Tensor], Tensor]]