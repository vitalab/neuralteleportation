import hashlib
import itertools
import json
import os
from pathlib import Path

import yaml

from neuralteleportation.utils.itertools import dict_values_product, listify_dict


def unravel_matrix_config(config_path: Path, output_dir: Path) -> int:
    with open(str(config_path), 'r') as stream:
        config = yaml.safe_load(stream)

    output_dir.mkdir(parents=True, exist_ok=True)

    single_run_config_count = 0

    # datasets
    for dataset_name in config["datasets"]:

        # models
        for model_name in config["models"]:
            # initalizers
            for initializer in config["initializers"]:
                # optimizers
                for optimizer_config in config["optimizers"]:

                    # teleport configuration
                    for teleport, teleport_config_kwargs in config["teleportations"].items():

                        # w/o teleport configuration
                        if teleport == "no_teleport":
                            # Ensure config collections are iterable, even if no config was defined
                            # This is done to simplify the generation of the configuration matrix
                            teleport_config_kwargs, teleport_mode_configs = {}, [("no_teleport", {})]

                        # w/ teleport configuration
                        else:  # teleport == "teleport"
                            # Copy the config to play around with its content without affecting the config loaded in memory
                            teleport_config_kwargs = teleport_config_kwargs.copy()

                            teleport_mode_obj = teleport_config_kwargs.pop("mode")
                            teleport_mode_configs = []
                            for teleport_mode, teleport_mode_config_kwargs in teleport_mode_obj.items():

                                # Ensure config collections are iterable, even if no config was defined
                                # This is done to simplify the generation of the configuration matrix
                                if teleport_mode_config_kwargs is None:
                                    teleport_mode_config_kwargs = {}

                                for teleport_mode_config_kwargs in dict_values_product(teleport_mode_config_kwargs):
                                    teleport_mode_configs.append((teleport_mode, teleport_mode_config_kwargs))

                        # generate matrix of training configuration
                        # (cartesian product of values for each training config kwarg)
                        teleport_configs = dict_values_product(teleport_config_kwargs)
                        config_matrix = itertools.product(teleport_configs, teleport_mode_configs)

                        # Iterate over different possible training configurations
                        for teleport_config_kwargs, (teleport_mode, teleport_mode_config_kwargs) in config_matrix:
                            single_run_config_count += 1
                            top_level_dict = {
                                "datasets": [dataset_name],
                                "models": [model_name],
                                "optimizers": [optimizer_config],
                                "initializers": [initializer],
                                "training_params": config["training_params"],
                                "teleportations": {teleport: None},
                                "runs_per_config": int(config["runs_per_config"]) if "runs_per_config" in config.keys() else 1
                            }
                            if teleport != "no_teleport":
                                top_level_dict["teleportations"][teleport] = {
                                    "mode": {teleport_mode: listify_dict(teleport_mode_config_kwargs)},
                                    **listify_dict(teleport_config_kwargs)
                                }

                            # Generate a unique name for the file, based on its content
                            # file with the same content should have the same name
                            config_hash = hashlib.sha1(json.dumps(top_level_dict, sort_keys=True).encode()).hexdigest()

                            with open(str(output_dir.joinpath(f"{config_hash}.yml")), 'w') as single_run_config_file:
                                yaml.dump(top_level_dict, single_run_config_file)

    return single_run_config_count


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Unravel a matrix configuration file into a directory of configuration files for single-run "
                    "experiments")
    parser.add_argument("config", type=Path,
                        help="Path to the YAML file describing the configuration matrix to unroll")
    parser.add_argument("--output_dir", type=Path, default=os.getcwd(),
                        help="Directory in which to save the generated single-run configurations")
    args = parser.parse_args()

    single_run_config_count = unravel_matrix_config(args.config, args.output_dir)

    print(f"Unraveled the matrix configuration file into {single_run_config_count} single-run configuration files, \n"
          f"saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
