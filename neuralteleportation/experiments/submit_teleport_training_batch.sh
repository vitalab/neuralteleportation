#!/bin/bash

GIT_CLONE_DIR=$1
EXPERIMENT_CONFIG_DIR=$2
EMAIL=$3

for experiment_config in $EXPERIMENT_CONFIG_DIR; do
  sbatch --mail-user="$EMAIL" --mail-type=ALL \
    ./submit_teleport_training.sh "$GIT_CLONE_DIR" "$experiment_config"
done
