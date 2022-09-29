import argparse
import csv
import os
import pandas as pd
import sys

# from SMART_CORE.safe_open import safe_open_w
from utils.safe_open import safe_open_w


def sort_StudentStep(outputFolder, course):
	# StudentStepRollUp = os.path.join(str(Path.home()), 'OneDrive/PASTEL Project/SMART/DataShop Export/OLI Biology/ALMAP spring 2014 DS 960 (Problem View fixed and Custom Field fixed)/ds1934_student_step_All_Data_3679_2020_0726_145039.txt') #nerve, kona, mocha
	# StudentStepRollUp = os.path.join(str(Path.home()), 'Downloads/ds1934_student_step_All_Data_3679_2020_0726_145039.txt') #nerve, kona, mocha
	if course == 'oli_intro_bio':
		StudentStepRollUp = 'SMART_CORE/OneDrive-2020-12-04/ds1934_student_step_All_Data_3679_2020_0726_145039.txt'  # OLI Introduction to Biology
		StudentStepRollUp_filename = os.path.split(StudentStepRollUp)[1]
		StudentStepRollUp_sorted = os.path.join(outputFolder, StudentStepRollUp_filename[:-4] + '_sorted.txt')
	elif course == 'oli_gen_chem':
		StudentStepRollUp = 'SMART_CORE/OneDrive-2020-12-04/ds4650_student_step_All_Data_6766_2021_0407_080036.txt'  # OLI General Chemistry I
		StudentStepRollUp_filename = os.path.split(StudentStepRollUp)[1]
		StudentStepRollUp_sorted = os.path.join(outputFolder, StudentStepRollUp_filename[:-4] + '_sorted.txt')
	else:
		print("Invalid input for course title.")
		sys.exit()

	if not os.path.isfile(StudentStepRollUp_sorted):
		print("Sorting StudentStep ({})...".format(course))
		
		with open(StudentStepRollUp, 'r') as read_file:
			csvreader = csv.reader(read_file, delimiter='\t')
			header = next(csvreader)
			sorted_rows = sorted(csvreader, key=lambda row: (row[2], row[7]))  # Student id, step start time

		# To limit the number of students to the top student_threshold number of students based on number of responses
		# per student
		student_threshold = 100
		df = pd.DataFrame(sorted_rows)
		unique_students = df[2].unique()
		response_counts = df[2].value_counts()[0:student_threshold]
		for student in unique_students:
			if student not in response_counts.index:
				df = df[(df[2] != student)]
		sorted_rows = df.to_numpy().tolist()

		with safe_open_w(StudentStepRollUp_sorted) as write_file:  # using a custom open function to create 'output' folder if does not exist
			csvwriter = csv.writer(write_file, delimiter='\t')
			csvwriter.writerow(header)
			
			# for all students
			csvwriter.writerows(sorted_rows)

		print("Done.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	outputFolder = 'output' 
	parser.add_argument('-outputFolder',  default=outputFolder, type=str)
	args = parser.parse_args()
	outputFolder = args.outputFolder  # string
	
	for course in ['oli_intro_bio', 'oli_gen_chem']:
		sort_StudentStep(outputFolder, course)
