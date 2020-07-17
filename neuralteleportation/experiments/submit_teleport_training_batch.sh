#!/bin/bash

usage()
{
  echo "Usage: submit_teleport_training_batch [ -d | --project_root_dir ] [ -c | --experiment_config_dir ] [ -m | --email ]"
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

for experiment_config_file in $experiment_config_dir; do
  sbatch --mail-user="$email" --mail-type=ALL \
    ./submit_teleport_training.sh -d "$project_root_dir" -c "$experiment_config_file"
done
