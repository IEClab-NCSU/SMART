import sys
import csv
from collections import defaultdict
import statistics
import sys

def remove_outlier_Student_KC(read_file):
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
	n_times = 3

	lower_bound = mean - n_times*sd
	upper_bound = mean + n_times*sd
	print('Lower bound: ', lower_bound)
	print('Upper bound: ', upper_bound)
	student_kc_outlier = set() #Any Student-KC with max_opportunity beyond the interval is an outlier

	for key, value in max_opportunity.items():
		if value < lower_bound or value > upper_bound:
			student_kc_outlier.add(key)
			print(key)

	with open(read_file, 'r') as f1:
		csvreader = csv.reader(f1, delimiter = ',')
		header = next(csvreader)

		write_file = read_file[:-4] + '_outliersRemoved_' + str(n_times) + 'sd.csv'
		with open(write_file, 'wb') as f2:
			csvwriter = csv.writer(f2, delimiter = ',')
			csvwriter.writerow(header)

			for row in csvreader:
				if not row:
					break
				if (row[2], row[KC_SMART_index]) not in student_kc_outlier:
					csvwriter.writerow(row)

			
	print(write_file)

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-StudentStep_file', type=str)

	# args = parser.parse_args()
	# StudentStep_file = args.StudentStep_file
	StudentStep_file = sys.argv[1]
	remove_outlier_Student_KC(StudentStep_file)