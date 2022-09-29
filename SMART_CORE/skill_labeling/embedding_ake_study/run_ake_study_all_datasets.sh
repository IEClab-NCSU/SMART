#!/bin/bash

# $1 - number of trials
# sh run_ake_study_all_datasets.sh <num_trials>

echo "Entire AKE study began at "
date

python prepare_directories.py

for dataset_name in inspec kdd oli-intro-bio oli-gen-chem  # inspec kdd nus krapivin pubmed oli-intro-bio oli-gen-chem
do
  for candidate_method in parsing ngrams # ngrams
  do
    sh run_ake_study.sh $dataset_name $candidate_method $1  > output/temp/$dataset_name-$candidate_method.txt 2>&1 &
  done
done

wait
echo "Entire AKE study completed at "
date