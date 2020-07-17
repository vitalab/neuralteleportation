#!/bin/bash

# Request resources --------------
#SBATCH --account def-pmjodoin
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=8          # Number of cores (not cpus)
#SBATCH --mem=32000M               # memory (per node)
#SBATCH --time=00-12:00            # time (DD-HH:MM)

GIT_CLONE_DIR=$1
EXPERIMENT_CONFIG_FILE=$2

# Navigate to where you git cloned the repository
cd "$GIT_CLONE_DIR" || {
  echo "Could not cd to directory: $GIT_CLONE_DIR"
  exit 1
}

# Activate shared virtual environment for the project
source "$HOME"/projects/def-pmjodoin/vitalab/virtualenv/neuralteleporation/bin/activate

# Make the code of the project visible to the python interpreter, without installing it in the environment.
# This is done to allow multiple users to use the same environment with different version of the project code.
export PYTHONPATH=$PYTHONPATH:$PWD

# Run task
python neuralteleporation/experiments/teleport_training.py --config "$EXPERIMENT_CONFIG_FILE"

# Exit normally on completion
exit
