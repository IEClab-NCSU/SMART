"""Parse the pool referenced Assessment xml files and return back to parent xml"
"""

from bs4 import BeautifulSoup
import os

def getPools(pools, xml_path_mapping):
	steps = []
	bodies = []
	for pool in pools:
		#print(pool)
		pool_xml = str(pool['idref'])+'.xml'
		#filepath = os.path.join(pool_directory, pool_xml)
		filepath = xml_path_mapping[pool_xml]
			
		if os.path.isfile(filepath):
			#print('filepath:', filepath)
		
			with open(filepath) as f:
				soup = BeautifulSoup(f, 'xml')
		
		else:
			print(filepath, ' does not exist')
			return [], []
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
				

	return steps, bodies

