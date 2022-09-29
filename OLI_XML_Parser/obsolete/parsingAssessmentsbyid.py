"""
Parse all the Assessment xml files and save them to a folder called "Assessments"

"""

from bs4 import BeautifulSoup
import os
from safe_open import safe_open_w
import csv
from pool import getPools
import re

def file_path_mapper(content):
	xml_path_mapping = dict()
	for root, directories, filenames in os.walk(content):
		for file in filenames:
			xml_path_mapping[file] = os.path.join(root, file)
	return xml_path_mapping

def parse(content, outputFolder):
	xml_path_mapping = file_path_mapper(content)
	tag_options = ['multiple_choice', 'question', 'short_answer', 'ordering']
	#print(xml_path_mapping['u3_biomacromolecules_unit_quiz_pool.xml'])
	with safe_open_w(os.path.join(outputFolder,'assessments.csv')) as fin: #modified open(). It opens file in wb mode by creating parent directories if needed
	#with open(outputFilename1, 'wb') as fin:
		writer = csv.writer(fin)

		count = 0
		for root, directories, filenames in os.walk(content):

			for directory in directories:
				if directory in ["x-oli-assessment2", "x-oli-inline-assessment", "high_stakes"]:
					workbookcontents = os.listdir(os.path.join(root, directory))
					if os.path.join(root, directory) == os.path.join(content,'x-ims-assessment/high_stakes'):
						print(directory)
						continue
					for file in workbookcontents:
						filepath = os.path.join(root, directory, file)

						if os.path.isfile(filepath) and file.endswith('.xml'):
							with open(filepath) as f:
								soup = BeautifulSoup(f, "xml")
							
						
							steps = []
							bodies = []
							for tag_option in tag_options:
								tags = soup.findAll(tag_option)
								for tag in tags:
									if tag.has_attr('id') and tag.body is not None:
										steps.append(tag['id'])

										body = tag.body.getText()
										if tag.name in ['multiple_choice', 'question']:
											responses = tag.findAll('response')
											maxScore = 0
											correctOptions = ''
											for response in responses:
												if float(response.get('score',0)) > maxScore:
													correctOptions = response.get('match','')
													maxScore = float(response.get('score',0))

											correctOptions = correctOptions.split(',')
											choices = tag.findAll('choice')
											for choice in choices:
												if choice['value'] in correctOptions:
													body += '\n' + choice.getText()

										# print(body)
										bodies.append(body)


							if len(steps) == 0:
								pools = soup.findAll("pool_ref")
								if pools:
									steps, bodies = getPools(pools, xml_path_mapping)
								else:
									continue
							#print('type(all)', type(all), 'type(tag)', type(tag))
							for step, body in zip(steps, bodies):
								#with open(outputFolder+os.path.basename(os.path.splitext(filepath)[0]) + '_' + t['id'] + '.txt', 'w+') as w:
									#w.write(que)
								###
								step = re.sub(r' +', '', step) #remove whitespaces
								
								#print(file, 'STEP...', step, 'BODY...',body)
								# que = body.get_text().encode('utf-8')
								
								# que = body.decode('utf-8')
								que = re.sub(r'\n', ' ', body) #replace newline with a space (' ')
								que = re.sub(r'\t', ' ', que) #replace tab with a space (' ')
								que = re.sub(r' +', ' ', que) #replace one or more spaces with a single space(' ')
								# que = re.sub(r'[^\w\d\s]', ' ', que) #replaces anything which is not [a-zA-Z0-9_], [0-9] or [ \t\n\r\f\v] with a single space (' ')
								
								
								###
								stepname = os.path.basename(os.path.splitext(filepath)[0]) + '_' + step
								row = [stepname, que]
								writer.writerow(row)
								
								count += 1
								#print os.path.basename(os.path.splitext(filepath)[0]) + '_' + t['id']

	print("The number of assessment items is... ", count)
	print("Parsing Assessments Done...")
