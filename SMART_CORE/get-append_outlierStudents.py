import argparse
import csv
from collections import defaultdict
import statistics
import os

def getAppend_outlierStudents(read_file):
	max_opportunity = defaultdict(int)
	with open(read_file, 'r') as f1:
		csvreader = csv.reader(f1, delimiter = ',')
		header = next(csvreader)
		KC_SMART_index = header.index('KC_SMART')
		for row in csvreader:
			if not row:
				break
			# row[2] is student id
			#row[KC_SMART_index] is KC_SMART
			#row[KC_SMART_index + 1] is KC_SMART's opportunity
			if int(row[KC_SMART_index + 1]) > max_opportunity[(row[2], row[KC_SMART_index])]:
				max_opportunity[(row[2], row[KC_SMART_index])] = int(row[KC_SMART_index + 1])

	mean = statistics.mean(max_opportunity.values())
	sd = statistics.pstdev(max_opportunity.values())

	print('mean: ', mean)
	print('Standard Deviation: ', sd)
	SD_multiplier = 3

	lower_bound = mean - SD_multiplier*sd
	upper_bound = mean + SD_multiplier*sd
	print('Lower bound: ', lower_bound)
	print('Upper bound: ', upper_bound)
	student_kc_outliers = set() #Any Student-KC with max_opportunity beyond the interval is an outlier

	
	#Get outliers
	new_student_outliers = set()
	for key, value in max_opportunity.items():
		if value < lower_bound or value > upper_bound:
			student_kc_outliers.add(key)
			new_student_outliers.add(key[0])
			print(key)

	print(new_student_outliers)
	print('No of outlier students: ', len(new_student_outliers))

	#Append outlierStudents file
	with open('student_outliers.csv', 'a') as f:
		csvwriter = csv.writer(f)
		for student in new_student_outliers:
			csvwriter.writerow([student])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-StudentStep_file', type=str)

	args = parser.parse_args()
	StudentStep_file = args.StudentStep_file

	getAppend_outlierStudents(StudentStep_file)
