"""
Parse XML files from OLI Biology course for workbook by page and save them to a folder called "Paragraphs"
"""

from bs4 import BeautifulSoup, Comment
import os
from pathlib import Path


def parse(content):
	count = 0
	for root, directories, filename in os.walk(content):
		for directory in directories:
			if directory == "x-oli-workbook_page":
				workbookcontents = os.listdir(os.path.join(root, directory))
				for file in workbookcontents:
					filepath = os.path.join(root, directory, file)
					if os.path.dirname(filepath) == 'content/x-oli-workbook_page':
						continue
					if os.path.isfile(filepath) and file.endswith('.xml'):
						with open(filepath) as f:
							soup = BeautifulSoup(f, "xml")

						for script in soup.find_all(['param', 'author']):
							script.extract()

						result = ""
						for child in soup.findAll(['head', 'body']):
							temp = child.get_text().encode('utf-8')
							result += temp.decode('utf-8')


						print(os.path.splitext(filepath)[0], result)

	print("The number of paragraphs is .. ", count)
	print("Parsing Workbook Done...")


if __name__ == '__main__':
	content = os.path.join(str(Path.home()),'Documents/OneDrive/PASTEL Project/SMART/OLI course data/intro_bio/content/') #my macbook
	
	parse(content)
