#!/bin/bash

__usage="
Usage: submit_teleport_training_batch.sh --project_root_dir PROJECT_ROOT_DIR
                                         --experiment_config_dir EXPERIMENT_CONFIG_DIR
                                         --email EMAIL

required arguments:
  --project_root_dir PROJECT_ROOT_DIR, -d PROJECT_ROOT_DIR
                        Root directory of the project's code
                        (typically where you cloned the repository)
  --experiment_config_dir EXPERIMENT_CONFIG_DIR, -c EXPERIMENT_CONFIG_DIR
                        Directory of the YAML configuration files to run as individual,
                        parallel jobs
                        This directory must not be deleted until all jobs have left
                        pending status!!! (the config files are only loaded once the
                        jobs are active)
  --email EMAIL, -m EMAIL
                        Email address at which SLURM will send notifications
                        regarding the states of the submitted jobs
"

usage()
{
  echo "$__usage"
  exit 2
}

PARSED_ARGUMENTS=$(getopt -n submit_teleport_training_batch -o d:c:m: --long project_root_dir:,experiment_config_dir:,email: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -d | --project_root_dir)        project_root_dir="$2"       ; shift 2 ;;
    -c | --experiment_config_dir)   experiment_config_dir="$2"  ; shift 2 ;;
    -m | --email)                   email="$2"                  ; shift 2 ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

required_args=("$project_root_dir" "$experiment_config_dir" "$email")
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

submit_config_dir() {
  # Submit individual jobs for all the configuration files in``experiment_config_dir``
  for experiment_config_file in "$experiment_config_dir"/*.yml; do
    sbatch --mail-user="$email" --mail-type=ALL \
      neuralteleportation/experiments/submit_teleport_training.sh -d "$project_root_dir" -c "$experiment_config_file"
  done
}

submit_config_file() {
  echo
}

submit_config_dir
