"""
Reduce rows of student step data for quick testing on a smaller dataset
"""

import sys
import csv
from collections import defaultdict
from safe_open import *
from datetime import datetime
import collections

def reduce_rows(file, n):
	rows = []	
	with open(file, 'r') as read_file:
		csvreader = csv.reader(read_file, delimiter=',')
		for index, row in enumerate(csvreader):
			if index == n:
				break
			rows.append(row)
			
	reduced_file = file[:-4] + '_' + str(n) + 'rows.csv'
	with open(reduced_file, 'wb') as write_file:
		writer = csv.writer(write_file)
		writer.writerows(rows)
	print(reduced_file)

if __name__ == '__main__':
	#file = '/Users/rajkumarshrivastava/OneDrive/SMART/DataShop Export/OLI Biology/ALMAP spring 2014 DS 960 (Problem View fixed and Custom Field fixed)/ds1934_student_step_All_Data_3679_2020_0414_091736.txt' #my macbook
	#file = 'StudentStepRollUp_KC_local.csv'
	file = sys.argv[1] #file to reduce rows
	count = sys.argv[2] #no. of rows to keep
	reduce_rows(file, int(count))
