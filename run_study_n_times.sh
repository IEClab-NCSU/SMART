#!/bin/bash

# This script takes in one parameter which specifies the outputFolder (e.g., /sdfsd/sdfsdf/ser/result_06_17__1724hrs)
# where all the files generated would be eventually saved.
# It calls run_study.sh n times.

rm -rf output
mkdir output
mkdir output/child_logs output/temp_logs
date
#python sort_StudentStep.py -outputFolder output
python SMART_CORE/sort_StudentStep.py
python SMART_CORE/download_models.py
date

echo "Started study at "
date
for n_run in 1 2 3 4 5
# for n_run in 1 2 3 4 5 6 7 8 9 10
do
	echo "Study iteration $n_run began at "
	date
	sh run_study.sh $n_run
	echo "Study iteration $n_run completed at "
	date
done

echo "All $n_run iterations completed at."
date

#rm output/ds1934_student_step_All_Data_3679_2020_0726_145039_sorted.txt

# Linux does not give conventional writing permission for files on OneDrive.
# For this, output.csv was created and appended locally.

#this file is now ready to be moved to OneDrive.
echo "Copying contents of output folder to $1"
mkdir $1 

if [ $? -ne 0 ] ; then
    echo "$1 already exists. Creating a copy of the output folder (appended with the current process id)."
    mkdir $1_$$
    cp -R output/* $1_$$
else
    echo "Created output folder without error."
    cp -R output/* $1
fi

echo "Please delete the output folder from local folder - manually"
#rm -r output
#rm output.csv.lock
echo "Completed study at "
date
