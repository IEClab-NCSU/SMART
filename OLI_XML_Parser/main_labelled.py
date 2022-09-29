"""Parses the xml files in the course repository and generates two csv files: assessments.csv and paragraphs.csv
"""
import argparse
from pathlib import Path
import os

from safe_open import safe_open_w
from parse_idrefs import parse_idrefs


def file_path_mapper(content):
	xml_path_mapping = dict()
	for root, directories, filenames in os.walk(content):
		for file in filenames:
			if file[-4:] == '.xml':
				xml_path_mapping[file] = os.path.join(root, file)
	return xml_path_mapping
		
if __name__ == '__main__':
	input = os.path.join(str(Path.home()),'OneDrive/SMART/OLI course data/gen_chem_v2_3') # local machine; OLI General Chemistry I
	# input = os.path.join(str(Path.home()),'OneDrive/SMART/OLI course data/intro_bio/')  # local machine; OLI Biology
	# input = os.path.join(str(Path.home()),'OneDrive/PASTEL Project/SMART/OLI course data/intro_bio/') # nerve
	
	outputFolder = os.path.join(str(Path.home()),'OneDrive/SMART/assessment_paragraph data/gen_chem') # local; OLI General Chemistry I
	# outputFolder = os.path.join(str(Path.home()),'OneDrive/SMART/assessment_paragraph data/intro_bio (with periods)_labelled/')  # local; OLI Biology
	# outputFolder = os.path.join(str(Path.home()),'OneDrive/PASTEL Project/SMART/assessment_paragraph data/intro_bio (with periods)_labelled/') # nerve
		
	#get command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-outputFolder', default = outputFolder, type=str)	
	args = parser.parse_args()
	outputFolder = args.outputFolder

	content = os.path.join(input, 'content')
	xml_path_mapping = file_path_mapper(content)

	#create output files
	with safe_open_w(os.path.join(outputFolder, 'paragraphs.csv'))as _:
		pass
	
	with safe_open_w(os.path.join(outputFolder, 'assessments.csv')) as _:
		pass

	# get the required version of the course contents by identifying the respective organization.xml file
	# organization = os.path.join(input, 'organizations/introucdavis/organization.xml') # for OLI Biology
	organization = os.path.join(input, 'organizations/default/organization.xml') # for OLI General Chemistry I

	# parse assessments and paragraphs based on the organization.xml file
	parse_idrefs(organization, xml_path_mapping, outputFolder)

	print(outputFolder)