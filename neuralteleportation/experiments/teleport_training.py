import copy
import itertools
import uuid
import math
from pathlib import Path

import comet_ml  # Must be imported before torch for annoying reasons
import torch
import torch.optim as optim
import yaml
from torch import nn
from torch.cuda import is_available as cuda_avail

from neuralteleportation.metrics import accuracy, accuracy_top5
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.experiment_run import run_model
from neuralteleportation.training.experiment_setup import get_model, get_dataset_subsets
from neuralteleportation.training.teleport import optim as teleport_optim
from neuralteleportation.training.teleport.optim import OptimalTeleportationTrainingConfig
from neuralteleportation.training.teleport.pseudo import PseudoTeleportationTrainingConfig
from neuralteleportation.training.teleport.pseudo import DistributionTeleportationTrainingConfig
from neuralteleportation.training.teleport.random import RandomTeleportationTrainingConfig
from neuralteleportation.utils.itertools import dict_values_product
from neuralteleportation.utils.logger import DiskLogger, MultiLogger, CometLogger

__training_configs__ = {"no_teleport": TrainingConfig,
                        "random": RandomTeleportationTrainingConfig,
                        "optim": OptimalTeleportationTrainingConfig,
                        "pseudo": PseudoTeleportationTrainingConfig,
                        "same_distr": DistributionTeleportationTrainingConfig}


def make_experiment(out_root: Path):
    assert out_root.exists()
    experiment_id = str(uuid.uuid4())  # Create a unique ID randomly
    experiment_dir = out_root / experiment_id
    assert not experiment_dir.exists()  # Check for collision (Extremely unlikely - I assume it will never happen)
    experiment_dir.mkdir()
    return experiment_dir, experiment_id


def run_experiment(config_path: Path, out_root: Path, data_root_dir: Path = None, save_weights=False,
                   enable_comet=False) -> None:
    with open(str(config_path), 'r') as stream:
        config = yaml.safe_load(stream)

    # Setup metrics to compute
    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy, accuracy_top5])

    # Get training params
    all_training_params = config["training_params"] if isinstance(config["training_params"], list) else [
        config["training_params"]]

    # datasets
    for dataset_name in config["datasets"]:
        dataset_kwargs = {}
        if data_root_dir is not None:
            dataset_kwargs.update(root=data_root_dir, download=False)
        train_set, val_set, test_set = get_dataset_subsets(dataset_name, **dataset_kwargs)
        for training_params in all_training_params:
            # models
            for model_obj in config["models"]:
                model_obj = copy.deepcopy(model_obj)
                model_kwargs = {}
                model_name = model_obj
                if not isinstance(model_obj, str):
                    model_name = model_obj.pop("cls")
                    model_kwargs = model_obj
                # initalizers
                for initializer in config["initializers"]:
                    config['initializer'] = initializer

                    # optimizers
                    for optimizer_kwargs in config["optimizers"]:
                        optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
                        optimizer_name = optimizer_kwargs.pop("cls")
                        lr_scheduler_kwargs = optimizer_kwargs.pop("lr_scheduler", None)
                        has_scheduler = False
                        if lr_scheduler_kwargs:
                            lr_scheduler_name = lr_scheduler_kwargs.pop("cls")
                            lr_scheduler_interval = lr_scheduler_kwargs.pop("interval", "epoch")
                            if "lr_lambda" in lr_scheduler_kwargs.keys():
                                # WARNING: Take care of what you pass in as lr_lambda as the string is directly
                                # evaluated
                                # This is needed to transform lambda functions defined as strings to a python callable
                                lr_scheduler_kwargs["lr_lambda"] = eval(lr_scheduler_kwargs.pop("lr_lambda"))

                            if "steps_per_epoch" in lr_scheduler_kwargs.keys():
                                steps = len(train_set) / training_params['batch_size']
                                lr_scheduler_kwargs['steps_per_epoch'] = math.floor(steps) if training_params['drop_last_batch'] else math.ceil(steps)

                            has_scheduler = True

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
                                # Copy the config to play around with its content without affecting the config loaded
                                #  in memory
                                teleport_config_kwargs = copy.deepcopy(teleport_config_kwargs)

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
                                num_runs = int(config["runs_per_config"]) if "runs_per_config" in config.keys() else 1
                                for _ in range(num_runs):
                                    experiment_path, experiment_id = make_experiment(out_root)

                                    if enable_comet:
                                        logger = MultiLogger([
                                            DiskLogger(experiment_path),
                                            CometLogger(experiment_id)
                                        ])
                                    else:
                                        logger = DiskLogger(experiment_path)

                                    training_config = training_config_cls(
                                        optimizer=(optimizer_name, optimizer_kwargs),
                                        lr_scheduler=(lr_scheduler_name,
                                                      lr_scheduler_interval,
                                                      lr_scheduler_kwargs) if has_scheduler else None,
                                        device='cuda' if cuda_avail() else 'cpu',
                                        logger=logger,
                                        **training_params,
                                        **teleport_config_kwargs,
                                        **teleport_mode_config_kwargs,
                                    )

                                    # Run experiment (setting up a new model and optimizer for each experiment)
                                    model = get_model(dataset_name, model_name, device=training_config.device,
                                                    initializer=initializer, **model_kwargs)
                                    optimizer = getattr(optim, optimizer_name)(model.parameters(), **optimizer_kwargs)
                                    lr_scheduler = None
                                    if has_scheduler:
                                        lr_scheduler = getattr(optim.lr_scheduler, lr_scheduler_name)(optimizer, **lr_scheduler_kwargs)
                                    run_model(model, training_config, metrics,
                                            train_set, test_set, val_set=val_set,
                                            optimizer=optimizer, lr_scheduler=lr_scheduler)

                                    if save_weights:
                                        torch.save(model.state_dict(), experiment_path / 'weights.pt')


def main():
    default_out_root = Path('./out')

    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Run an arbitrary series of experiments training neural networks using teleportations")
    parser.add_argument("config", type=Path,
                        help="Path to the YAML file describing the configuration matrix of the experiments to run")
    parser.add_argument("--data_root_dir", type=Path,
                        help="Root directory of data inside which each dataset creates its own directory. "
                             "This option is useful in case the datasets must be pre-downloaded to a known location "
                             "(e.g. when working on a cluster with no internet access).")
    parser.add_argument("--out_root_dir", type=Path, default=default_out_root,
                        help="Root directory where the outputs of the training will be stored (e.g. metrics).")
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--enable_comet", action="store_true")
    args = parser.parse_args()

    # Manage output directory (for metrics)
    if args.out_root_dir == default_out_root:
        print(f'WARNING: Writing outputs (metrics) in {default_out_root}. You should probably set --out_root_dir.')
    print(f'INFO: Using output root dir: {args.out_root_dir}')
    args.out_root_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(args.config, data_root_dir=args.data_root_dir, out_root=args.out_root_dir,
                   save_weights=args.save_weights, enable_comet=args.enable_comet)


if __name__ == '__main__':
    main()
