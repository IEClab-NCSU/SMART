"""
Parse xml files for workbook folders by paragraph and save them to a folder called "Paragraphs"

"""
from bs4 import BeautifulSoup, Comment
import os


def parse():
    count = 0
    for root, directories, filename in os.walk('content/'):
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

                        for index, child in enumerate(soup.findAll(['p', 'ol', 'ul'])):
                            result = child.get_text().encode('utf-8')

                            with open('Paragraphs/'+os.path.basename(os.path.splitext(filepath)[0]) + str(index+1) + '.txt', 'w+') as w:
                                w.write(result)
                                count += 1

    print "The number of paragraphs is .. ", count
    print "Parsing Workbook Done..."
