import csv
import argparse
import sys

def reduce_StudentStep_KC(StudentStep_file):
	all_student_outliers = set()
	with open("student_outliers.csv", 'r') as f:
		csvreader = csv.reader(f)
		for row in csvreader:
			if not row:
				break
			all_student_outliers.add(row[0])


	with open(StudentStep_file, 'r') as f1:
		csvreader = csv.reader(f1, delimiter = ',')
		header = next(csvreader)

		#write_file = StudentStep_file[:-4] + '_outliersRemoved_' + str(SD_multiplier) + 'sd.csv'
		write_file = StudentStep_file[:-4] + '_outliersRemoved.csv'
		with open(write_file, 'wb') as f2:
			csvwriter = csv.writer(f2, delimiter = ',')
			csvwriter.writerow(header)

			for row in csvreader:
				if not row:
					break
				if row[2] not in all_student_outliers:
					csvwriter.writerow(row)
				
		print(write_file)

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-StudentStep_file', type=str)

	# args = parser.parse_args()
	# StudentStep_file = args.StudentStep_file

	StudentStep_file = sys.argv[1]

	reduce_StudentStep_KC(StudentStep_file)