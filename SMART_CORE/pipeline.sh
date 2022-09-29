#!/bin/bash

# This script performs the following tasks:
# 1. Run SMART
# 2. Update KC, Opportunity columns in Student Step csv file
# 3. Run R script to find model fit
# 4. Append output.csv with a new row for the model fit results

export CUDA_VISIBLE_DEVICES=""
echo "Running SMART...$6 $1 $2 $3 $5 $7"
python -u run_smart.py -strategyType $1 -encodingType $2 -clusteringType $3 -clusters $5 -n_run $4 -course $6 -merge $7 -outputFolder output/run_$4/$6/$1/$2/$3/$5_$7| tee -a output/temp_logs/$6SMART_stdout$$.txt
assessment_skill_mapping=$(cat output/temp_logs/$6SMART_stdout$$.txt | tail -n 1) #last line

echo "Filling KC, Opportunity column...$6 $1 $2 $3 $5 $7"
python -u SMART_CORE/get_StudentStep_KC_Opportunity_withThreshold.py -assessment_skill_mapping $assessment_skill_mapping -course $6 | tee -a output/temp_logs/$6SMART_StudentStep_stdout$$.txt
StudentStep_KC_Opportunity_outFile=$(cat output/temp_logs/$6SMART_StudentStep_stdout$$.txt | tail -n 1) #last line

#optional
# echo "Reducing Student step rows"
# python -u reduce_row.py $StudentStep_KC_Opportunity_outFile 1000 | tee -a logs/reduce_row_stdout$$.txt
# StudentStep_KC_Opportunity_outFile_reduced=$(cat logs/reduce_row_stdout$$.txt | tail -n 1) #last line

echo "Running Rscript...$6 $1 $2 $3 $5 $7"
# Rscript preprocess_runAFM.R $StudentStep_KC_Opportunity_outFile | tee -a output/temp_logs/R_stdout$$.txt
Rscript SMART_CORE/model_fit/preprocess_runAFM_RMSE_1fold.R $StudentStep_KC_Opportunity_outFile | tee -a output/temp_logs/$6R_stdout$$.txt

echo "Running compute_outputRow.py...$6 $1 $2 $3 $5 $7"
python SMART_CORE/compute_outputRow.py -strategyType $1 -encodingType $2 -clusteringType $3 -clusters $5 -R_output output/temp_logs/$6R_stdout$$.txt -n_run $4 -course $6 -merge $7 #-StudentStep_KC_Opportunity_outFile $StudentStep_KC_Opportunity_outFile

#remove the processes associated to this process id only
rm -rf output/temp_logs/$6SMART_stdout$$.txt output/temp_logs/$6SMART_StudentStep_stdout$$.txt output/temp_logs/$6R_stdout$$.txt
