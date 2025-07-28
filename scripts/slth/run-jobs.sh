#!/bin/zsh

# Specify hyperparameters
datasets=("2iris" "3iris" "2wine" "3wine")
models=("BVQC" "VQC" "NN" "SNN")

# Exit if sbatch command not available
if ! command -v sbatch > /dev/null; then
  echo "\033[1;31mYou are not in a slurm environment. Exiting...\033[0m"
  exit 1
fi

for dataset in ${datasets[@]}; do
  for model in ${models[@]}; do

    # skip BVQC if dataset is iris or wine, as they have 3 classes and thus are not suitable for binary classification
    if [[ "$model" == "BVQC" && ("$dataset" == "3iris" || "$dataset" == "3wine") ]]; then
      continue
    fi

    job_name="s-${dataset:0:2}-${model:0:1}"
    sbatch --job-name="${job_name}" scripts/slth/job.sh $dataset $model
  done
done