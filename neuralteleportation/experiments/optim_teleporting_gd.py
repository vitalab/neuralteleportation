from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from typing import Tuple, Callable, Any

# Necessary to import Comet first to use Comet's auto logging facility and
# to avoid "Please import comet before importing these modules" error.
# (see ref: https://www.comet.ml/docs/python-sdk/warnings-errors/)
import comet_ml  # noqa
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TeleportationTrainingConfig

ModelEvalFunc = Callable[[NeuralTeleportationModel, Tensor, Tensor, TrainingMetrics, "TeleportationTrainingConfig"],
                         Number]


@dataclass
class OptimalTeleportationTrainingConfig(TeleportationTrainingConfig):
    num_teleportations: int = 10
    num_batches: int = 1
    comparison_metric: Tuple[ModelEvalFunc, Callable[[Any, Any], bool]] = None  # Required


def teleport_model_to_optimize_metric(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics,
                                      config: OptimalTeleportationTrainingConfig) -> nn.Module:
    print(f"Selecting best of {config.num_teleportations} random COBs "
          f"w.r.t. {config.comparison_metric[0].__name__}")

    # Extract a single batch on which to compute gradients for each model to be compared
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    data, target = [], []
    for (data_batch, target_batch), _ in zip(dataloader, range(config.num_batches)):
        data.append(data_batch)
        target.append(target_batch)
    data = torch.stack(data).to(device=config.device)
    target = torch.stack(target).to(device=config.device)

    # NOTE: The input shape passed to `NeuralTeleportationModel` must take into account the batch dimension
    model = NeuralTeleportationModel(network=model, input_shape=(2,) + config.input_shape)

    # Unpack the configuration for the metric to use to optimize gradients
    metric_func, metric_compare = config.comparison_metric

    optimal_metric = metric_func(model, data, target, metrics, config)
    model.cpu()  # Move model to CPU to avoid having 2 models on the GPU (to avoid possible CUDA OOM error)
    optimal_model = model

    for _ in range(config.num_teleportations):
        teleported_model = deepcopy(model).random_teleport(cob_range=config.cob_range,
                                                           sampling_type=config.cob_sampling)
        teleported_model.to(config.device)  # Move model back to chosen device before computing gradients
        metric = metric_func(teleported_model, data, target, metrics, config)
        teleported_model.cpu()  # Move model back to CPU after computation is done (to avoid possible CUDA OOM error)
        if metric_compare(metric, optimal_metric):
            optimal_model = teleported_model
            optimal_metric = metric

    return optimal_model.network.to(config.device)


def main():
    import operator
    from pathlib import Path

    from torch import nn

    from neuralteleportation.changeofbasisutils import get_available_cob_sampling_types
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.config import TrainingMetrics
    from neuralteleportation.training.experiment_run import run_model
    from neuralteleportation.training.experiment_setup import get_cifar10_models, get_cifar10_datasets
    from neuralteleportation.utils.logger import init_comet_experiment
    from neuralteleportation.utils.model_eval import weighted_grad_norm, loss_lookahead_diff

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    cob_ranges = [0.7, 1.2]
    cob_samplings = get_available_cob_sampling_types()
    teleport_every_n_epochs = [1, 2, 5, 10]
    for sampling_type in cob_samplings:
        for cob_range in cob_ranges:
            for n in teleport_every_n_epochs:
                for comparison_metric in [(weighted_grad_norm, operator.gt),
                                          (loss_lookahead_diff, operator.gt)]:
                    for model in get_cifar10_models(device='cuda'):
                        env_name = "{}_{}_optim_teleport_{}_{}_every_{}".format(model.__class__.__name__,
                                                                                comparison_metric[0].__name__,
                                                                                sampling_type, cob_range, n)
                        print("Starting: ", env_name)
                        config = OptimalTeleportationTrainingConfig(
                            input_shape=(3, 32, 32),
                            device='cuda',
                            cob_range=cob_range,
                            cob_sampling=sampling_type,
                            teleport_every_n_epochs=n,
                            epochs=20,
                            comparison_metric=comparison_metric,
                            comet_logger=init_comet_experiment(Path(".comet.config")),
                        )
                        run_model(teleport_model_to_optimize_metric, model,
                                  config, metrics,
                                  cifar10_train, cifar10_test, val_set=cifar10_val)


if __name__ == '__main__':
    main()
