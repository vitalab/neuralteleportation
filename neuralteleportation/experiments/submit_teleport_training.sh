#!/bin/bash

# Request resources --------------
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=8          # Number of cores (not cpus)
#SBATCH --mem=32000M               # memory (per node)
#SBATCH --time=00-02:00            # time (DD-HH:MM)

__usage="
Usage: submit_teleport_training.sh --project_root_dir PROJECT_ROOT_DIR
                                   --dataset_dir DATASET_DIR
                                   --experiment_config_file EXPERIMENT_CONFIG_FILE
                                   [--out_root_dir OUT_ROOT_DIR]

required arguments:
  --project_root_dir PROJECT_ROOT_DIR, -p PROJECT_ROOT_DIR
                        Root directory of the project's code
                        (typically where you cloned the repository)
  --dataset_dir DATASET_DIR, -d DATASET_DIR
                        Directory where pre-downloaded datasets are stored
  --experiment_config_file EXPERIMENT_CONFIG_FILE, -c EXPERIMENT_CONFIG_FILE
                        Path of the YAML configuration file to run as a single,
                        sequential job
                        This file must not be deleted until the job has left
                        pending status!!! (the config file is only loaded once
                        the job is active)

optional arguments:
  --out_root_dir OUT_ROOT_DIR
                        Root directory under which local results generated by
                        the experiment will be stored
                        (in a subfolder created for the experiment)
"

usage()
{
  echo "$__usage"
  exit 2
}

PARSED_ARGUMENTS=$(getopt -n submit_teleport_training -o p:d:c: --long project_root_dir:,dataset_dir:,experiment_config_file:,out_root_dir: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -p | --project_root_dir) project_root_dir="$2"; shift 2 ;;
    -d | --dataset_dir) dataset_dir="$2"; shift 2 ;;
    -c | --experiment_config_file) experiment_config_file="$2"; shift 2 ;;
    -o | --out_root_dir) out_root_dir="$2"; shift 2 ;; # FIXME Using -o does not work but the long form does... why?
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

# Additional manual checks on arguments
required_args=("$project_root_dir" "$dataset_dir" "$experiment_config_file")
for required_arg in "${required_args[@]}"; do
  if [ -z "$required_arg" ]
    then
      echo "Missing one or more required argument(s)"; usage
  fi
done

# Ensure the project's root directory exists
if [ ! -d "$project_root_dir" ]; then
  echo "Provided project root directory does not exist: $project_root_dir."; exit 1
fi

check_timeout()
{
  if [ $? -eq 124 ]; then
    echo "Creating/installing virtualenv timed out"; exit 124
  fi
}

# Install and activate a virtual environment directly on the compute node
module load httpproxy # To allow connections to Comet server
module load python/3.7
timeout 3m virtualenv --no-download "$SLURM_TMPDIR"/env
check_timeout
source "$SLURM_TMPDIR"/env/bin/activate
timeout 4m pip install --no-index -r "$project_root_dir"/requirements/computecanada_wheel.txt
check_timeout
timeout 3m pip install "$project_root_dir"/.
check_timeout

# Copy any dataset we might use to the compute node
dataset_dir=${dataset_dir%/}/  # Ensure the presence of a trailing slash if there isn't one (to ensure consistent behavior with rsync)
compute_node_data_dir="$SLURM_TMPDIR"/data
rsync -a "$dataset_dir" "$compute_node_data_dir"

# If out_root_dir is not set, use ~/scratch/neuralteleportation
out_root=${out_root_dir:-~/scratch/neuralteleportation}

# Run task
python "$project_root_dir"/neuralteleportation/experiments/teleport_training.py "$experiment_config_file" \
  --data_root_dir "$compute_node_data_dir" --out_root_dir="$out_root"
