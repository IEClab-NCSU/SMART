import argparse
import csv
from collections import defaultdict
import os
from sort_StudentStep import sort_StudentStep
import sys


def clipOffUnderscore(step):
	pos = step.rfind('_')
	if pos != -1:
		step = step[:pos]
	return step


def fill_KC_Opportunity(assessment_skill_file, KC_model_name, outputFolder, course):
	assessment_skill_map = defaultdict()
	with open(assessment_skill_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			assessment_skill_map[row[0]] = row[1]

	# StudentStepRollUp = '/Users/rajkumarshrivastava/OneDrive/SMART/DataShop Export/OLI Biology/ALMAP spring 2014 DS 960 (Problem View fixed and Custom Field fixed)/ds1934_student_step_All_Data_3679_2020_0414_091736_sorted.txt' #my macbook
	if course == 'oli_intro_bio':
		StudentStepRollUp_sorted = os.path.join(outputFolder, 'ds1934_student_step_All_Data_3679_2020_0726_145039' + '_sorted.txt')  # OLI Biology
	elif course == 'oli_gen_chem':
		StudentStepRollUp_sorted = os.path.join(outputFolder, 'ds4650_student_step_All_Data_6766_2021_0407_080036' + '_sorted.txt')  # OLI General Chemistry I
		
	# create a sorted StudentStepRollUp if it does not exist.
	if not os.path.isfile(StudentStepRollUp_sorted):
		sort_StudentStep(outputFolder, course)

	observations = []
	with open(StudentStepRollUp_sorted, 'r') as read_file:
		csvreader = csv.reader(read_file, delimiter='\t')
		header = next(csvreader)

		KC_model_name = 'KC ('+KC_model_name+')'
		KC_model_index = header.index(KC_model_name)
		header[KC_model_index] = 'KC_SMART'  # changed name of KC model for RScript grep function.

		observations.append(header)

		Student_id_index = header.index('Anon Student Id')
		current_student_id = ''

		KC_opportunity_to_observationsCount = defaultdict(int)  # (KC, opportunity) -> #observations
		for index, row in enumerate(csvreader):
			# if index = 1000: #studentstep subset
			if not row:
				break

			assessment = row[4] + '_' + row[6]
			trimmed_assessment = clipOffUnderscore(assessment)

			while trimmed_assessment not in assessment_skill_map.keys() and trimmed_assessment != assessment:
				assessment = trimmed_assessment
				trimmed_assessment = clipOffUnderscore(assessment)

			if trimmed_assessment in assessment_skill_map.keys():
				skill = assessment_skill_map[trimmed_assessment]
			else:
				skill = 'UNKNOWN'

			row[KC_model_index] = skill

			if row[Student_id_index] != current_student_id:
				skill_to_opportunity = defaultdict(int)  # reset skill_to_opportunity
				current_student_id = row[Student_id_index]

			skill_to_opportunity[skill] += 1
			row[KC_model_index + 1] = skill_to_opportunity[skill]

			observations.append(row)

			KC_opportunity_to_observationsCount[(skill, skill_to_opportunity[skill])] += 1

		# end of for
	# file closed
	# Added 6/24/2021: Used to analyse the skill, opportunity, and opportunity count information for a run.
	'''Opportunity_Counts = assessment_skill_file[:-4] + 'KC_Opportunity_Count.csv'
	with open(Opportunity_Counts, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for skill in skill_to_opportunity.keys():
			for opportunity in range(1, skill_to_opportunity[skill]):
				count = KC_opportunity_to_observationsCount[(skill, opportunity)]
				new_row = [skill, opportunity, count]
				writer.writerow(new_row)'''

	StudentStep_KC_Opportunity = assessment_skill_file[:-4] + '_StudentStep_KC_Opportunity.csv'
	with open(StudentStep_KC_Opportunity, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')

		writer.writerow(observations[0])  # write header
		
		for row in observations[1:]:
			KC, opportunity = row[KC_model_index], row[KC_model_index+1]
			# Write row only if #observations for the opportunity is greater than equal to half of the first opportunity
			if KC_opportunity_to_observationsCount[(KC, opportunity)] >= KC_opportunity_to_observationsCount[(KC, 1)]/2:
				writer.writerow(row)
		
	# print the created file path for the next script to read
	print(StudentStep_KC_Opportunity)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-assessment_skill_mapping', type=str)
	parser.add_argument('-course', default='oli_intro_bio', type=str)

	args = parser.parse_args()

	assessment_skill_mapping_file = args.assessment_skill_mapping  # filename along with path
	course = args.course.lower()
	if course == 'oli_intro_bio':
		KC_model_name = 'Model1_clst100w_nm_nmfC10old-PV-models'  # change this for a different KC model (OLI Biology).
	elif course == 'oli_gen_chem':
		KC_model_name = 'ImportTest'  # change this for a different KC model (OLI General Chemistry I).
	else:
		print("Invalid input for course title.")
		sys.exit()
	outputFolder = 'output'
	
	fill_KC_Opportunity(assessment_skill_mapping_file, KC_model_name, outputFolder, course)
