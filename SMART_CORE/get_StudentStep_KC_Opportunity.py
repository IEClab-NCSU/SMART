"""
Run Local Python Files
Command to be used for running -

#echo "Enter assessment_skill_mapping_file";
#read assessment_skill_mapping_file

python get_StudentStep_KC_Opportunity.py $assessment_skill_mapping_file

For example -
python get_StudentStep_KC_Opportunity.py /Users/rajkumarshrivastava/OneDrive/SMART/Raj_data_2020/Results/assessment_skill_first_tf_1_assessment.csv
"""

import argparse
import csv
from collections import defaultdict
from safe_open import *
from datetime import datetime
import collections
import os
from sort_StudentStep import sort_StudentStep
import shutil

def clipOffUnderscore(step):
	pos = step.rfind('_')
	if pos !=-1:
		step = step[:pos]
	return step

def fill_KC_Opportunity(assessment_skill_file, KC_model_name, outputFolder):
	assessment_skill_map = defaultdict()
	with open(assessment_skill_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			assessment_skill_map[row[0]] = row[1]

	#StudentStepRollUp = '/Users/rajkumarshrivastava/OneDrive/SMART/DataShop Export/OLI Biology/ALMAP spring 2014 DS 960 (Problem View fixed and Custom Field fixed)/ds1934_student_step_All_Data_3679_2020_0414_091736_sorted.txt' #my macbook
	StudentStepRollUp_sorted = os.path.join(outputFolder, 'ds1934_student_step_All_Data_3679_2020_0726_145039' + '_sorted.txt')
		
	#create a sorted StudentStep if it does not exist.
	if not os.path.isfile(StudentStepRollUp_sorted):
		sort_StudentStep(outputFolder)

	with open(StudentStepRollUp_sorted, 'r') as read_file:
		csvreader = csv.reader(read_file, delimiter='\t')
		header = next(csvreader)
		
		#head_tail = os.path.split(assessment_skill_file)
		#StudentStep_KC_Opportunity = os.path.join(head_tail[0], n_run+'_'+head_tail[1][:-4] + '_StudentStep_KC_Opportunity.csv')
		StudentStep_KC_Opportunity = assessment_skill_file[:-4] + '_StudentStep_KC_Opportunity.csv'
		with open(StudentStep_KC_Opportunity, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			
			KC_model_name = 'KC ('+KC_model_name+')'
			KC_model_index = header.index(KC_model_name)
			header[KC_model_index] = 'KC_SMART' #changed name of KC model for RScript grep function.
			
			writer.writerow(header)
			
			Student_id_index = header.index('Anon Student Id')
			current_student_id = ''
			for index, row in enumerate(csvreader):
				# if index = 1000: #studentstep subset
				# 	break

				if not row:
					break

				assessment = row[4] + '_' + row[6]
				trimmed_assessment = clipOffUnderscore(assessment)

				while trimmed_assessment not in assessment_skill_map.keys() and trimmed_assessment != assessment:
					assessment = trimmed_assessment
					trimmed_assessment = clipOffUnderscore(assessment)
				
				if trimmed_assessment in assessment_skill_map.keys():
					skill = assessment_skill_map[trimmed_assessment]
					#key = trimmed_assessment
				else:
					skill = ''
					pass
			
				row[KC_model_index] = skill
				
				if row[Student_id_index] != current_student_id:
					opportunity_counter = collections.defaultdict(int)
					current_student_id = row[Student_id_index]

				opportunity_counter[skill] += 1
				row[KC_model_index + 1] = opportunity_counter[skill]

				writer.writerow(row)
				#print(index+1)
					
	# StudentStep_KC_Opportunity = assessment_skill_file[:-4] + '_StudentStep_KC_Opportunity.txt'
	# with safe_open_w(StudentStep_KC_Opportunity) as file:
	# 	file.writelines(','.join(i) + '\n' for i in rows_toWrite)

	# StudentStep_KC_Opportunity_csv = assessment_skill_file[:-4] + '_StudentStep_KC_Opportunity.csv'
	# with safe_open_wb(StudentStep_KC_Opportunity_csv) as write_file:
	# 	csvwriter = csv.writer(write_file)
	# 	csvwriter.writerows(rows_toWrite)

	# #Creating a tab-delimited (.txt) version of the above file	
	# print("Converting to tab-delimited file...")
	# with open(StudentStep_KC_Opportunity_csv) as inputFile:
	# 	StudentStep_KC_Opportunity_txt = StudentStep_KC_Opportunity_csv[:-4] + '.txt'
	# 	with open(StudentStep_KC_Opportunity_txt, 'w') as outputFile:
	# 		reader = csv.DictReader(inputFile, delimiter=',')
	# 		writer = csv.DictWriter(outputFile, reader.fieldnames, delimiter=',')
	# 		writer.writeheader()
	# 		writer.writerows(reader)
	# print (outputFilename3)

	print(StudentStep_KC_Opportunity)
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-assessment_skill_mapping', type=str)

	args = parser.parse_args()

	assessment_skill_mapping_file = args.assessment_skill_mapping #filename along with path
	KC_model_name = 'Model1_clst100w_nm_nmfC10old-PV-models' #change this for a different KC model.
	outputFolder = 'output'
	
	fill_KC_Opportunity(assessment_skill_mapping_file, KC_model_name, outputFolder)
