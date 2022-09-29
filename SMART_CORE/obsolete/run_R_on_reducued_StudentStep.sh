#!/bin/bash
#!/bin/sh
#This script computes the time taken by RScript and the trend with increasing n value.
echo "Started at "
date
for n in 1000 5000 20000 50000 100000 200000
do
	echo date	
	echo "Running for n = "$n
	python -u reduce_row.py $1 $n| tee temp.txt #$1 is StudentStep_KC_Opportunity_outFile
	StudentStep_KC_Opportunity_outFile_reduced=$(cat temp.txt | tail -n 1) #last line

	Rscript preprocess_runAFM.R $StudentStep_KC_Opportunity_outFile_reduced > R_output.txt
	rm -rf temp.txt

	python -u compute_Routput_row.py -StudentStep_KC_Opportunity_outFile $StudentStep_KC_Opportunity_outFile_reduced -R_output R_output.txt -n_rows $n| tee temp2.txt
	outputFolder=$(cat temp2.txt | tail -n 1) #last line
done

#Linux does not give conventional writing permission for files on OneDrive.
#For this, output.csv was created and appended locally.
#this file is now ready to be moved to OneDrive.
echo "Completed at "
date
mv R_output.csv $outputFolder
