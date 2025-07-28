#!/bin/bash

# Specify hyperparameters
datasets=("2iris" "3iris" "2wine" "3wine")
models=("BVQC" "VQC" "NN" "SNN")
pruning_techniques=("ONE_SHOT" "ITERATIVE")

for dataset in ${datasets[@]}; do
  for model in ${models[@]}; do

    # skip BVQC if dataset is iris or wine, as they have 3 classes and thus are not suitable for binary classification
    if [[ "$model" == "BVQC" && ("$dataset" == "3iris" || "$dataset" == "3wine") ]]; then
      continue
    fi

    for pruning_technique in ${pruning_techniques[@]}; do
      python ../../src/plot_wlth.py $dataset $model $pruning_technique
    done
  done
done