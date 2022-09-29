"""
Parse all the Assessment xml files and save them to a folder called "Assessments"

"""

from bs4 import BeautifulSoup
import os


def parse():
    count = 0
    for root, directories, filenames in os.walk('content/'):

        for directory in directories:
            if directory == "x-oli-assessment2" or directory == "x-oli-assessment2-pool" or \
                            directory == "x-oli-inline-assessment" or directory == "high_stakes":
                workbookcontents = os.listdir(os.path.join(root, directory))
                if os.path.join(root, directory) == 'content/x-ims-assessment/high_stakes':
                    print directory
                    continue
                for file in workbookcontents:
                    filepath = os.path.join(root, directory, file)

                    if os.path.isfile(filepath) and file.endswith('.xml'):
                        with open(filepath) as f:
                            soup = BeautifulSoup(f, "xml")
                        all_questions = soup.findAll('body')

                        if all_questions is None:
                            continue
                        tag = soup.multiple_choice

                        if tag is not None and tag.has_attr('id'):
                            tag_name = 'multiple_choice'

                        else:
                            tag = soup.question
                            if tag is not None:
                                file_name = tag['id']
                                tag_name = 'question'
                            else:
                                continue
                        tag = soup.findAll(tag_name)

                        for question, t in zip(all_questions, tag):
                            que = question.get_text().encode('utf-8')
                            with open('Assessments/'+os.path.basename(os.path.splitext(filepath)[0]) + '_' + t['id'] + '.txt', 'w+') as w:
                                w.write(que)
                            with open(os.path.splitext(filepath)[0] + '_' + t['id'] + '.txt', 'w+') as w:
                                w.write(que)
                            count += 1
                            #print os.path.basename(os.path.splitext(filepath)[0]) + '_' + t['id']

    print "The number of assessment items is... ", count
    print "Parsing Assessments Done..."
