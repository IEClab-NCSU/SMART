#!/bin/bash

# $1 - dataset_name $2 - candidate_method $3 - num_trials
# sh run_ake_study.sh <num_trials>

echo "Began AKE Study on $1 dataset with $2 candidate keyphrase generation for $3 trials."
date
python -u run_keyphrase_extractor_study.py -datasetName $1 -candidateMethod $2 -numberTrials $3

wait
echo "Completed AKE Study on $1 dataset with $2 candidate keyphrase generation for $3 trials. "
date
