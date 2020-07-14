import random
from dataclasses import dataclass

# Necessary to import Comet first to use Comet's auto logging facility and
# to avoid "Please import comet before importing these modules" error.
# (see ref: https://www.comet.ml/docs/python-sdk/warnings-errors/)
import comet_ml  # noqa
from torch import nn
from torch.utils.data import Dataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TeleportationTrainingConfig


@dataclass
class RandomTeleportationTrainingConfig(TeleportationTrainingConfig):
    teleport_prob: float = 1.  # Always teleport by default when reaching `teleport_every_n_epochs`


def teleport_model_randomly(model: NeuralTeleportationModel, train_dataset: Dataset,
                            metrics: TrainingMetrics, config: RandomTeleportationTrainingConfig) -> nn.Module:
    if random.random() < config.teleport_prob:
        print("Applying random COB to model in training")
        model.random_teleport(
            cob_range=config.cob_range, sampling_type=config.cob_sampling)
    else:
        print("Skipping COB")

    return model


def main():
    from pathlib import Path

    from neuralteleportation.changeofbasisutils import get_available_cob_sampling_types
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.experiment_run import run_model
    from neuralteleportation.training.experiment_setup import get_cifar10_models, get_cifar10_datasets
    from neuralteleportation.utils.logger import init_comet_experiment

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    cob_ranges = [0.7, 1.2]
    cob_samplings = get_available_cob_sampling_types()
    teleport_every_n_epochs = [1, 2, 5, 10]
    for sampling_type in cob_samplings:
        for cob_range in cob_ranges:
            for n in teleport_every_n_epochs:
                for model in get_cifar10_models(device='cuda'):
                    env_name = "{}_teleport_{}_{}_every_{}".format(model.__class__.__name__,
                                                                   sampling_type, cob_range, n)
                    print("Starting: ", env_name)
                    config = RandomTeleportationTrainingConfig(
                        input_shape=(3, 32, 32),
                        device='cuda',
                        cob_range=cob_range,
                        cob_sampling=sampling_type,
                        teleport_every_n_epochs=n,
                        epochs=20,
                        comet_logger=init_comet_experiment(Path(".comet.config")),
                    )
                    model = NeuralTeleportationModel(network=model, input_shape=(2,) + config.input_shape)
                    run_model(teleport_model_randomly, model,
                              config, metrics,
                              cifar10_train, cifar10_test, val_set=cifar10_val)


if __name__ == '__main__':
    main()
