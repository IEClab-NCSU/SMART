import argparse
import csv
from collections import defaultdict
from safe_open import *
from datetime import datetime
import collections
from pathlib import Path
import os

def clipOffUnderscore(step):
	pos = step.rfind('_')
	if pos !=-1:
		step = step[:pos]
	return step

def checkMapping(assessments):
	assessment_set = set()
	with open(assessments, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			assessment_set.add(row[1])

	# StudentStepRollUp = os.path.join(str(Path.home()), 'OneDrive/SMART/DataShop Export/OLI Biology/ALMAP spring 2014 DS 960 (Problem View fixed and Custom Field fixed)/ds1934_student_step_All_Data_3679_2020_0726_145039.txt') #local - OLI Biology
	StudentStepRollUp = os.path.join(str(Path.home()), 'OneDrive/SMART/DataShop Export/OLI General Chemistry I/ds4650_student_step_All_Data_6766_2021_0407_080036.txt') #local - OLI Chemistry
	mappings = 'mappings.csv'

	with open(StudentStepRollUp, 'r') as read_file:
		csvreader = csv.reader(read_file, delimiter='\t')

		with open(mappings, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
								
			for index, row in enumerate(csvreader):
				if not row:
					break

				assessment = row[4] + '_' + row[6]
				trimmed_assessment = clipOffUnderscore(assessment)

				while trimmed_assessment not in assessment_set and trimmed_assessment != assessment:
					assessment = trimmed_assessment
					trimmed_assessment = clipOffUnderscore(assessment)
				
				found = False
				if trimmed_assessment in assessment_set:
					found = True
							
				rowTowrite = [row[4], row[6], found]

				writer.writerow(rowTowrite)
				#print(index+1)
	
	print(mappings)
	
	
if __name__ == '__main__':	
	# assessments = os.path.join(str(Path.home()), "OneDrive/SMART/assessment_paragraph data/intro_bio (with periods)_labelled/assessments.csv") #local - OLI Biology
	assessments = os.path.join(str(Path.home()), "OneDrive/SMART/assessment_paragraph data/gen_chem/assessments.csv") #local - OLI Chemistry

	checkMapping(assessments)
