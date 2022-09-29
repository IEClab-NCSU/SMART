#!/bin/bash

# Run the SMART pipeline (pipeline.sh) for different combinations of hyperparameters

# $1 - n_run
for course in oli_intro_bio oli_gen_chem
do
  for strategy in assessment paragraph #strategyTypes: assessment, paragraph
  do
    for encoding in bert s-bert #encodingTypes: tf, tfidf, bert, s-bert
    # for encoding in bert #encodingTypes: tf, tfidf
    do
      for clustering in first #clusteringTypes: first, second
      # for clustering in first #clusteringTypes: first, second
      do
        for num_clusters in 10 50 100 150 200 #num_clusters:  'None' for iterative or integer1, integer2, ..., integerN
        do
          for merge_status in on #merge_status: 'on' or 'off'
          do
            echo "Generating results for $course $strategy $encoding $clustering $num_clusters $merge_status"
            sh SMART_CORE/pipeline.sh $strategy $encoding $clustering $1 $num_clusters $course $merge_status > output/child_logs/$1$course$strategy$encoding$clustering$num_clusters$merge_status.txt 2>&1 &
            #sh generateResult.sh $strategy $encoding $clustering $1
          done
        done
      done
    done
  done
done

wait 
echo "Combinations done."

