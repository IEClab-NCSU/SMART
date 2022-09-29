"""
Parse xml files for workbook folders by paragraph and save them to a folder called "Paragraphs"

"""
from safe_open import safe_open_w
import csv
import os
from bs4 import BeautifulSoup, Comment
import re

def parse(content, outputFolder):
	with safe_open_w(os.path.join(outputFolder,'paragraphs.csv')) as fin: #modified open(). It opens file in wb mode by creating parent directories if needed
	#with open(outputFilename1, 'wb') as fin:
		writer = csv.writer(fin)

		count = 0
		for root, directories, filename in os.walk(content):
			for directory in directories:
				if directory == "x-oli-workbook_page":
					workbookcontents = os.listdir(os.path.join(root, directory))
					for file in workbookcontents:
						filepath = os.path.join(root, directory, file)
						if os.path.dirname(filepath) == os.path.join(content,'x-oli-workbook_page'):
							continue
						if os.path.isfile(filepath) and file.endswith('.xml'):
							with open(filepath) as f:
								soup = BeautifulSoup(f, "xml")

							for script in soup.find_all(['param', 'author']):
								script.extract()

							for index, child in enumerate(soup.findAll(['p', 'ol', 'ul'])):
								result = child.get_text().encode('utf-8')
								###
								result = result.decode('utf-8')
								result = re.sub(r'\n', ' ', result) #replace newline with a space (' ')
								result = re.sub(r'\t', ' ', result) #replace tab with a space (' ')
								result = re.sub(r' +', ' ', result) #replace one or more spaces with a single space(' ')
								# result = re.sub(r'[^\w\d\s]', ' ', result) #replaces anything which is not [a-zA-Z0-9_], [0-9] or [ \t\n\r\f\v] with a single space (' ')
								###
								if len(result) > 1:
									row = [os.path.basename(os.path.splitext(filepath)[0]) + str(index+1), result]
									writer.writerow(row)
									count += 1

								# with open('Paragraphs_6_24/'+os.path.basename(os.path.splitext(filepath)[0]) + str(index+1) + '.txt', 'w+') as w:
								# 	w.write(result)
								#	

	print("The number of paragraphs is .. ", count)
	print("Parsing Workbook Done...")
