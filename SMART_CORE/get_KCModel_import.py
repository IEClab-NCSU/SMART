"""
Create a Datashop KC Model Import file by creating a KC column in the DataShop KC Model Export file.
"""

import sys
import csv
from collections import defaultdict
from utils.safe_open import safe_open_w
from pathlib import Path
import os
from datetime import datetime

def trimFromLast_(step):
	pos = step.rfind('_')
	if pos !=-1:
		step = step[:pos]
	return step

def addSkillColumn(assessment_skill_file, DataShopExport, KC_model_name, outputFolder):
	assessment_skill_map = defaultdict()
	with open(assessment_skill_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			assessment_skill_map[row[0]] = row[1]
	# print(assessment_skill_map['inline_m4_telophase_DIGT_q1'])
	# return
	with open(DataShopExport, 'r') as read_file:
		csvreader = csv.reader(read_file, delimiter='\t')

		rows_toWrite = []
		for index, row in enumerate(csvreader):
			if not row:
				break
			if index == 0:
				skill = KC_model_name
				#key = 'prob+step'
			else: 
				assessment = row[2] + '_' + row[4]
				trimmed_assessment = trimFromLast_(assessment)

				while trimmed_assessment not in assessment_skill_map.keys() and trimmed_assessment != assessment:
					assessment = trimmed_assessment
					trimmed_assessment = trimFromLast_(assessment)
				
				if trimmed_assessment in assessment_skill_map.keys():
					skill = assessment_skill_map[trimmed_assessment]
					#key = trimmed_assessment
				else:
					skill = ''
					#key = ''
			# print(skill)
			rows_toWrite.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], skill])
			#print(index+1)

	export_base = os.path.basename(DataShopExport)[:-4]
	assessment_skill_base = os.path.basename(assessment_skill_file)[:-4]
	DataShopImport = os.path.join(outputFolder, export_base + assessment_skill_base + '.txt') #nerve
	# DataShopImport = '/Users/rajkumarshrivastava/OneDrive/SMART/Raj_data_2020/July14_1544hrs/Model1_clst100w_nm_nmfC10old-PV-models_Import.txt' #my macbook
			
	with safe_open_w(DataShopImport) as write_file:
		csvwriter = csv.writer(write_file, delimiter='\t')
		csvwriter.writerows(rows_toWrite)

	print (DataShopImport)
	
# Driver code	
if __name__ == '__main__':
	# assessment_skill_file = '/mnt/wwn-0x5000c500af70d9d3-part1/SMART/02_25_2021_tf/run_1/assessment/tf/first/200/1_assessment_skill_compound_first_tf_200_assessment.csv'
	# assessment_skill_file = '/mnt/wwn-0x5000c500af70d9d3-part1/SMART/02_24_2021 bert tfidf/run_1/assessment/bert/first/200/1_assessment_skill_compound_first_bert_200_assessment.csv'
	# assessment_skill_file = '/mnt/wwn-0x5000c500af70d9d3-part1/SMART/02_24_2021 bert tfidf/run_1/assessment/tfidf/first/200/1_assessment_skill_compound_first_tfidf_200_assessment.csv'
	assessment_skill_file = os.path.join(str(Path.home()), 'OneDrive/SMART/SMART output/Champion_Output/gen_chem/Hyperparameter Tuning/5_assessment_skill_first_tfidf_133_assessment.csv')
	# assessment_skill_file = 'output/_assessment_skill_first_bert_200_assessment.csv'
	DataShopExport = os.path.join(str(Path.home()),'OneDrive/SMART/DataShop Import/gen_chem/gen_chem_final_import.txt') #nerve
	# DataShopExport = os.path.join(str(Path.home()),'Documents/OneDrive/SMART/DataShop Export/OLI Biology/ALMAP spring 2014 DS 960 (Problem View fixed and Custom Field fixed)/Model1_clst100w_nm_nmfC10old-PV-models.txt') #macbook
	
	today_date  = datetime.today().strftime('%m_%d_%Y') #today's date for naming convention

	KC_model_name = 'KC(SMART_champion_merge)'

	outputFolder = 'DataShop_Import_' + today_date #create a new folder with today's date
		
	addSkillColumn(assessment_skill_file, DataShopExport, KC_model_name, outputFolder)
