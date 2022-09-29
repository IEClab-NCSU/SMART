import csv
from collections import defaultdict
if __name__ == '__main__':
	with open('DataShop_Student_Step_rollup.csv', 'r') as f:
		old_csv = csv.reader(f)
		old_stepId = []
		old_stepName = []
		for index, row in enumerate(old_csv):
			if not row:
				break
			old_stepId.append(row[0])
			if "\\xef\\xbb\\xbfStep ID" in row[0]:
				print('found==',index)
			old_stepName.append(row[4])
	with open('Model1_clst100w_nm_nmfC10old-PV-models.txt', 'r') as f:
		new_csv = csv.reader(f, delimiter = '\t')
		new_stepId = []
		new_stepName = []
		for index, row in enumerate(new_csv):
			if not row:
				break
			new_stepId.append(row[0])
			# if 'Step ID' in row[0]:
			# 	print('found==',index)
			new_stepName.append(row[4])

	with open('Original KC.txt', 'r') as f:
		original_KC_csv = csv.reader(f, delimiter = '\t')
		original_KC_stepId = []
		original_KC_stepName = []
		for index, row in enumerate(original_KC_csv):
			if not row:
				break
			original_KC_stepId.append(row[0])
			# if 'Step ID' in row[0]:
			# 	print('found==',index)
			original_KC_stepName.append(row[4])
	
	print('# Total stepIds in old file', len(old_stepId))
	print('# Total stepIds in new file', len(new_stepId))

	print('# Distinct StepIds in old file', len(set(old_stepId)))
	print('# Distinct StepIds in new file', len(set(new_stepId)))

	print('# Total step names in old file', len(old_stepName))
	print('# Total step names in new file', len(new_stepName))

	print('# Distinct StepNames in old file', len(set(old_stepName)))
	print('# Distinct StepNames in new file', len(set(new_stepName)))

	print('Are set of step ids same:', set(old_stepId) == set(new_stepId))
	print('Are set of step names same:', set(old_stepName) == set(new_stepName))

	#print('Stepnames only in old_file: ', set(old_stepName) - set(new_stepName))
	#print('Stepnames only in new_file: ', len(set(new_stepName) - set(old_stepName)))
	print('StepIds only in old_file: ', set(old_stepId) - set(new_stepId))
	print('StepIds only in new_file: ', set(new_stepId) - set(old_stepId))
	
	#common_stepIds = set(old_stepId).intersection(set(new_stepId))
	#print('# Common step ids: ', len(common_stepIds))
	

	# common_stepnames = set(old_stepName).intersection(set(new_stepName))
	# print('# Common step names: ', len(common_stepnames))

	# common_stepIds = set(old_stepId).intersection(set(new_stepId))
	# print('# Common stepIds: ', len(common_stepIds))
	'''
	old_dict = defaultdict(set)
	for i in range(len(old_stepId)):
		old_dict[old_stepId[i]].add(old_stepName[i])
	for key, value in old_dict.items():
		print(len(value))
		if len(value) > 1:
			print(value)
	'''

