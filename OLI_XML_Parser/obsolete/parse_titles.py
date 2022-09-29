"""
Parse all the Assessment xml files and save them to a folder called "Assessments"
"""

from bs4 import BeautifulSoup
import os
from safe_open import safe_open_wb
import csv
from pool import getPools
import re


def parse_titles(content):
	count = 0
	titles = []
	with open('titles.csv', 'wb') as f:
		writer = csv.writer(f)
		for root, directories, filenames in os.walk(content):
			for directory in directories:
				workbookcontents = os.listdir(os.path.join(root, directory))
				for file in workbookcontents:
					filepath = os.path.join(root, directory, file)
					if os.path.isfile(filepath) and file.endswith('.xml'):
						with open(filepath) as f:
							soup = BeautifulSoup(f, "xml")
						title_tags = soup.findAll('title')
						for title_tag in title_tags:
							title = title_tag.get_text().encode('utf-8')
							
							title = title.decode('utf-8')
							title = re.sub(r'\n', ' ', title) #replace newline with a space (' ')
							title = re.sub(r'\t', ' ', title) #replace tab with a space (' ')
							title = re.sub(r' +', ' ', title) #replace one or more spaces with a single space(' ')
							title = re.sub(r'[^\w\d\s]', ' ', title) #replaces anything which is not [a-zA-Z0-9_], [0-9] or [ \t\n\r\f\v] with a single space (' ')
							
							writer.writerow([title])
							count += 1
											

	print "Total number of titles... ", count
	print "Parsing Assessments Done..."

if __name__ == '__main__':
	content = '/Users/rajkumarshrivastava/OneDrive/SMART/Text Mining for Skill Discovery (Abhishek 2017)/Introduction To Biology/content' #my macbook
	parse_titles(content)
