#!/bin/bash

# Request resources --------------
#SBATCH --account def-pmjodoin
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=8          # Number of cores (not cpus)
#SBATCH --mem=32000M               # memory (per node)
#SBATCH --time=00-12:00            # time (DD-HH:MM)

usage()
{
  echo "Usage: submit_teleport_training [ -d | --project_root_dir ] [ -c | --experiment_config_file ]"
  exit 2
}

PARSED_ARGUMENTS=$(getopt -n submit_teleport_training -o d:c: --long project_root_dir:,experiment_config_file: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -d | --project_root_dir)        project_root_dir="$2"       ; shift 2 ;;
    -c | --experiment_config_file)  experiment_config_file="$2"  ; shift 2 ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

required_args=("$project_root_dir" "$experiment_config_file")
for required_arg in "${required_args[@]}"; do
  if [ -z "$required_arg" ]
    then
      echo "Missing one or more required argument(s)"; usage
  fi
done

# Navigate to the project's root directory
cd "$project_root_dir" || {
  echo "Could not cd to directory: $project_root_dir. Verify that it exists."
  exit 1
}

# Install and activate a virtual environment directly on the compute node
module load python/3.7
module load scipy-stack # For scipy, matplotlib and pandas
virtualenv --no-download "$SLURM_TMPDIR"/env
source "$SLURM_TMPDIR"/env/bin/activate
pip install --no-index -r requirements/computecanada_wheel.txt
pip install .

# Copy any dataset we might use to the compute node
dataset_shared_dir="$HOME"/projects/def-pmjodoin/vitalab/datasets/neuralteleportation/
compute_node_data_dir="$SLURM_TMPDIR"/data
rsync -a "$dataset_shared_dir" "$compute_node_data_dir"

# Run task
python neuralteleportation/experiments/teleport_training.py --config "$experiment_config_file" \
  --data_root_dir "$compute_node_data_dir"
