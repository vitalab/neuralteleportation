from copy import deepcopy
from typing import Callable, Sequence

from torch import nn
from torchvision.datasets import VisionDataset

from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.training.training import test


def run_single_output_training(train_fct: Callable, models: Sequence[nn.Module],
                               config: TrainingConfig, metrics: TrainingMetrics,
                               train_set: VisionDataset, test_set: VisionDataset,
                               val_set: VisionDataset = None):
    for model in models:
        print(f"Training {model.__class__.__name__}")
        trained_model = train_fct(model, train_dataset=train_set, metrics=metrics, config=config, val_dataset=val_set)
        print("Testing {}: {} \n".format(model.__class__.__name__, test(trained_model, test_set, metrics, config)))
        print()


def run_multi_output_training(train_fct: Callable, models: Sequence[nn.Module],
                              config: TrainingConfig, metrics: TrainingMetrics,
                              train_set: VisionDataset, test_set: VisionDataset,
                              val_set: VisionDataset = None):
    for model in models:
        print(f"Training {model.__class__.__name__}")
        trained_models = train_fct(model, train_dataset=train_set, metrics=metrics, config=deepcopy(config),
                                   val_dataset=val_set)
        for id, trained_model in trained_models.items():
            print("Testing {}: {} \n".format(id, test(trained_model, test_set, metrics, config)))
        print()
