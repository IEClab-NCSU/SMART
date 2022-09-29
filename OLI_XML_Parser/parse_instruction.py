"""
Recuarsively crawl and parse xml files and distinguish them as assessment or paragraph.
"""
from bs4 import BeautifulSoup
import csv
import os
import re

from pool import getPools
from safe_open import safe_open_w

def regex_filters(string):
	string = re.sub(r'\n', ' ', string) #replace newline with a space (' ')
	string = re.sub(r'\t', ' ', string) #replace tab with a space (' ')
	string = re.sub(r'(\. )+', ' _ ', string) #replace blanks, i.e., '. . . .' with '_'
	string = re.sub(r'_+', '_', string) #replace one or more underscores with a single underscore ('_')
	string = re.sub(r' +', ' ', string) #replace one or more spaces with a single space(' ')
	string = re.sub(r'\.+', '.', string) #replace one or more periods with a single period('.')
	# string = re.sub(r'[^\w\d\s]', ' ', string) #replaces anything which is not [a-zA-Z0-9_], [0-9] or [ \t\n\r\f\v] with a single space (' ')	
	return string
	
def parse_paragraph(soup, unit_id, filename, outputFolder):
	with open(os.path.join(outputFolder,'paragraphs.csv'), 'a') as f_out: #modified open(). It opens file in wb mode by creating parent directories if needed
		writer = csv.writer(f_out)
		for index, child in enumerate(soup.findAll(['p', 'ol', 'ul'])):
			result = child.get_text()
			# result = child.get_text().encode('utf-8')
			# result = result.decode('utf-8')

			result = regex_filters(result)

			if len(result) > 1:
				row = [unit_id, filename[:-4] + str(index+1), result]
				writer.writerow(row)

def parse_assessment(soup, unit_id, filename, outputFolder, xml_path_mapping):
	with open(os.path.join(outputFolder,'assessments.csv'), 'a') as f_out: #modified open(). It opens file in wb mode by creating parent directories if needed
		writer = csv.writer(f_out)
	
		steps = []
		bodies = []
		# for tag_option in ['multiple_choice', 'question', 'short_answer', 'ordering']:
		# Note: 'fill_in_the_blank', 'numeric', and 'text' are necessary additions for General Chemistry I.
		for tag_option in ['multiple_choice', 'question', 'ordering', 'short_answer', 'fill_in_the_blank', 'numeric', 'text']:
			tags = soup.findAll(tag_option)
			for tag in tags:
				if tag.has_attr('id') and tag.body is not None:
					steps.append(tag['id'])

					body = tag.body.getText()

					responses = tag.findAll('response')
					maxScore = 0
					correctResponse = ''
					correctRespone_feedback = ''
					for response in responses:
						if float(response.get('score',0)) > maxScore:
							correctResponse = response.get('match','')
							maxScore = float(response.get('score',0))
							correctRespone_feedback = response.feedback
					correctOptions = correctResponse.split(',')
					choices = tag.findAll('choice')
					for choice in choices:
						if choice['value'] in correctOptions:
							body += '\n' + choice.getText()

					if correctRespone_feedback:
						body += '\n' + correctRespone_feedback.getText()
						

					hints = tag.findAll("hint")
					for hint in hints:
						body += '\n' + hint.getText()

					explanations = tag.findAll("explanation")
					for explanation in explanations:
						body += '\n' + explanation.getText()

					# # Remove the occurences of correct in assessments
					# body = body.replace('Correct', '')
					# body = body.replace('correct', '')

					bodies.append(body)

		# if this xml file itself does not contain any assessment item
		if len(steps) == 0:
			pools = soup.findAll("pool_ref")
			if pools:
				steps, bodies = getPools(pools, xml_path_mapping)

		for step, body in zip(steps, bodies):
			step = re.sub(r' +', '', step) #remove whitespaces
			stepname = filename[:-4] + '_' + step
			
			body = regex_filters(body)
			
			row = [unit_id, stepname, body]
			writer.writerow(row)
			
def parse_instruction(unit_id, filename, outputFolder, xml_path_mapping):	
	if filename not in xml_path_mapping:
		return
		
	file_path = xml_path_mapping[filename]	
	with open(file_path) as f:
		soup = BeautifulSoup(f, "xml")

	# tags used to reference to other xml files: xref, inline, activity, activity_link
	xrefs = soup.findAll(["xref"])
	inlines = soup.findAll(["inline"])
	activities = soup.findAll(["activity"])
	activity_links = soup.findAll(["activity_link"])
	

	if soup.findAll("workbook_page"):
		parse_paragraph(soup, unit_id, filename, outputFolder)
	else:
		parse_assessment(soup, unit_id, filename, outputFolder, xml_path_mapping)
	
	del xml_path_mapping[filename]

	#parse all the referenced xml files from the current assessment/paragraph xml file.

	for xref in xrefs:
		filename = xref['page'] + '.xml'
		if filename in xml_path_mapping:
			parse_instruction(unit_id, filename, outputFolder, xml_path_mapping)
	
	for inline in inlines:
		filename = inline['idref'] + '.xml'
		if filename in xml_path_mapping:
			parse_instruction(unit_id, filename, outputFolder, xml_path_mapping)
	
	for activity in activities:
		filename = activity['idref'] + '.xml'
		if filename in xml_path_mapping:
			parse_instruction(unit_id, filename, outputFolder, xml_path_mapping)
	
	for activity_link in activity_links:
		filename = activity_link['idref'] + '.xml'
		if filename in xml_path_mapping:
			parse_instruction(unit_id, filename, outputFolder, xml_path_mapping)

#code testing
if __name__ == '__main__':
	file_path = 'testing_parser/test_file.xml'	
	with open(file_path) as f:
		soup = BeautifulSoup(f, "xml")

	parse_assessment(soup, unit_id='something', filename=None, outputFolder='', xml_path_mapping='')