import argparse
from safe_open import safe_open_w
from parse_idrefs import parse_idrefs
from parsingWorkbook_by_paragraph import parse as parse_workbook
from parsingAssessmentsbyid import parse as parse_assessments
#from parsingAssessmentsbyid_old import parse as parse_assessments
from pathlib import Path
import os

def file_path_mapper(content):
	xml_path_mapping = dict()
	for root, directories, filenames in os.walk(content):
		for file in filenames:
			if file[-4:] == '.xml':
				xml_path_mapping[file] = os.path.join(root, file)
	return xml_path_mapping
		
if __name__ == '__main__':
	input = os.path.join(str(Path.home()),'Documents/OneDrive/PASTEL Project/SMART/OLI course data/intro_bio/') #my macbook
	# input = os.path.join(str(Path.home()),'OneDrive/PASTEL Project/SMART/OLI course data/intro_bio/') #nerve
	
	outputFolder = os.path.join(str(Path.home()),'Documents/OneDrive/PASTEL Project/SMART/assessment_paragraph data/intro_bio (with periods)_labelled/') #my macbook
	# outputFolder = os.path.join(str(Path.home()),'OneDrive/PASTEL Project/SMART/assessment_paragraph data/intro_bio (with periods)_labelled/') #nerve
		
	parser = argparse.ArgumentParser()
	parser.add_argument('-outputFolder', default = outputFolder, type=str)
	args = parser.parse_args()
	outputFolder = args.outputFolder

	content = os.path.join(input, 'content')
	xml_path_mapping = file_path_mapper(content)


	file = '_u1_m2_respiratory_inline01' + '.xml'
	print(xml_path_mapping[file])

	# parse_assessments(input_content, outputFolder)
	# parse_workbook(input_content, outputFolder)

