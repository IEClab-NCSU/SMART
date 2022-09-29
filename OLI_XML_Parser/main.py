import argparse
from parsingWorkbook_by_paragraph import parse as parse_workbook
from parsingAssessmentsbyid import parse as parse_assessments
#from parsingAssessmentsbyid_old import parse as parse_assessments
from pathlib import Path
import os

if __name__ == '__main__':
	# input_content = os.path.join(str(Path.home()),'Documents/OneDrive/PASTEL Project/SMART/OLI course data/intro_bio/content/') #my macbook
	input_content = os.path.join(str(Path.home()),'OneDrive/PASTEL Project/SMART/OLI course data/intro_bio/content/') #nerve
	
	# outputFolder = os.path.join(str(Path.home()),'Documents/OneDrive/PASTEL Project/SMART/assessment_paragraph data/intro_bio (with periods)/') #my macbook
	outputFolder = os.path.join(str(Path.home()),'OneDrive/PASTEL Project/SMART/assessment_paragraph data/intro_bio (with periods)/') #nerve
		
	parser = argparse.ArgumentParser()
	parser.add_argument('-outputFolder', default = outputFolder, type=str)
	args = parser.parse_args()
	outputFolder = args.outputFolder


	parse_assessments(input_content, outputFolder)
	parse_workbook(input_content, outputFolder)

	#parse_workbook(content)
	#parse_assessments(content)
