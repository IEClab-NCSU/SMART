import sys
import csv
from collections import defaultdict
import statistics
import sys

def remove_outlier_Opportunities(read_file):
	max_opportunity = 50

	with open(read_file, 'r') as f1:
		csvreader = csv.reader(f1, delimiter = ',')
		header = next(csvreader)
		KC_SMART_index = header.index('KC_SMART')

		write_file = read_file[:-4] + '_OpportunityoutliersRemoved_' + str(max_opportunity) + '.csv'
		with open(write_file, 'wb') as f2:
			csvwriter = csv.writer(f2, delimiter = ',')
			csvwriter.writerow(header)

			for row in csvreader:
				if not row:
					break
				if int(row[KC_SMART_index+1]) <= max_opportunity:
					csvwriter.writerow(row)

			
	print(write_file)

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-StudentStep_file', type=str)

	# args = parser.parse_args()
	# StudentStep_file = args.StudentStep_file
	StudentStep_KC_file = sys.argv[1]
	remove_outlier_Opportunities(StudentStep_KC_file)