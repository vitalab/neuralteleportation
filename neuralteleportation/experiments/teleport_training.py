import itertools
from pathlib import Path

# Necessary to import Comet first to use Comet's auto logging facility and
# to avoid "Please import comet before importing these modules" error.
# (see ref: https://www.comet.ml/docs/python-sdk/warnings-errors/)
import comet_ml  # noqa
import torch.optim as optim
import yaml
from torch import nn

from neuralteleportation.metrics import accuracy
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.experiment_run import run_model
from neuralteleportation.training.experiment_setup import get_model, get_dataset_subsets
from neuralteleportation.training.teleport import optim as teleport_optim
from neuralteleportation.training.teleport.optim import OptimalTeleportationTrainingConfig
from neuralteleportation.training.teleport.random import RandomTeleportationTrainingConfig
from neuralteleportation.utils.itertools import dict_values_product
from neuralteleportation.utils.logger import init_comet_experiment

__training_configs__ = {"no_teleport": TrainingConfig,
                        "random": RandomTeleportationTrainingConfig,
                        "optim": OptimalTeleportationTrainingConfig}


def run_experiment(config_path: Path, comet_config: Path, data_root_dir: Path = None) -> None:
    with open(str(config_path), 'r') as stream:
        config = yaml.safe_load(stream)

    # Setup metrics to compute
    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # datasets
    for dataset_name in config["datasets"]:
        dataset_kwargs = {}
        if data_root_dir is not None:
            dataset_kwargs.update(root=data_root_dir, download=False)
        train_set, val_set, test_set = get_dataset_subsets(dataset_name, **dataset_kwargs)

        # models
        for model_name in config["models"]:

            # optimizers
            for optimizer_kwargs in config["optimizers"]:
                optimizer_name = optimizer_kwargs.pop("cls")

                # teleport configuration
                for teleport, teleport_config_kwargs in config["teleportations"].items():

                    # w/o teleport configuration
                    if teleport == "no_teleport":
                        training_config_cls = __training_configs__["no_teleport"]
                        # Ensure config collections are iterable, even if no config was defined
                        # This is done to simplify the generation of the configuration matrix
                        teleport_config_kwargs, teleport_mode_configs = {}, [(training_config_cls, {})]

                    # w/ teleport configuration
                    else:  # teleport == "teleport"
                        # Copy the config to play around with its content without affecting the config loaded in memory
                        teleport_config_kwargs = teleport_config_kwargs.copy()

                        teleport_mode_obj = teleport_config_kwargs.pop("mode")
                        teleport_mode_configs = []
                        for teleport_mode, teleport_mode_config_kwargs in teleport_mode_obj.items():
                            training_config_cls = __training_configs__[teleport_mode]
                            if teleport_mode == "optim":
                                teleport_mode_config_kwargs["optim_metric"] = [
                                    getattr(teleport_optim, metric) for metric
                                    in teleport_mode_config_kwargs.pop("metric")
                                ]

                            # Ensure config collections are iterable, even if no config was defined
                            # This is done to simplify the generation of the configuration matrix
                            if teleport_mode_config_kwargs is None:
                                teleport_mode_config_kwargs = {}

                            for teleport_mode_single_config_kwargs in dict_values_product(teleport_mode_config_kwargs):
                                teleport_mode_configs.append((training_config_cls, teleport_mode_single_config_kwargs))

                    # generate matrix of training configuration
                    # (cartesian product of values for each training config kwarg)
                    teleport_configs = dict_values_product(teleport_config_kwargs)
                    config_matrix = itertools.product(teleport_configs, teleport_mode_configs)

                    # Iterate over different possible training configurations
                    for teleport_config_kwargs, (training_config_cls, teleport_mode_config_kwargs) in config_matrix:
                        training_config = training_config_cls(
                            optimizer=(optimizer_name, optimizer_kwargs),
                            device='cuda',
                            comet_logger=init_comet_experiment(comet_config),
                            epochs=20,  # TODO Move from hardcoded to config by dataset/model pair
                            **teleport_config_kwargs,
                            **teleport_mode_config_kwargs,
                        )

                        # Run experiment (setting up a new model and optimizer for each experiment)
                        model = get_model(dataset_name, model_name, device=training_config.device)
                        optimizer = getattr(optim, optimizer_name)(model.parameters(), **optimizer_kwargs)
                        run_model(model, training_config, metrics,
                                  train_set, test_set, val_set=val_set,
                                  optimizer=optimizer)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Run an arbitrary series of experiments training neural networks using teleportations")
    parser.add_argument("config", type=Path,
                        help="Path to the YAML file describing the configuration matrix of the experiments to run")
    parser.add_argument("--comet_config", type=Path, default=Path(".comet.config"),
                        help="Path to the Comet config file indicating how to log the experiments")
    parser.add_argument("--data_root_dir", type=Path,
                        help="Root directory of data inside which each dataset creates its own directory. "
                             "This option is useful in case the datasets must be pre-downloaded to a known location "
                             "(e.g. when working on a cluster with no internet access).")
    args = parser.parse_args()

    run_experiment(args.config, args.comet_config, data_root_dir=args.data_root_dir)


if __name__ == '__main__':
    main()
