import os
import re
from nltk import sent_tokenize

# create a list of workbook contents
def create_workbook_lists(use_sentence):
    workbook = []
    workbookname=[]
    directory = 'Paragraphs/'
    workbookcontents = os.listdir(directory)
    for file in workbookcontents:
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath) and file.endswith('.txt'):
            with open(filepath) as f:
                content = f.read()

            content = content.decode('utf-8')
            content = re.sub(r'\n', ' ', content)
            content = re.sub(r'\t', ' ', content)
            content = re.sub(r' +', ' ', content)

            if use_sentence:
                workbook.extend(sent_tokenize(content))

            else:
                workbook.append(content)
                workbookname.append(os.path.basename(filepath))

    #print workbook
    return workbook, workbookname


# Create a list of assessment items
def create_assessments_list():
    assessment = []
    assessmentname = []
    for root, directories, filename in os.walk('content/'):
        for directory in directories:
            if directory == "x-oli-assessment2" or directory == "x-oli-assessment2-pool" or \
                            directory == "x-oli-inline-assessment" or directory == "high_stakes":
                workbookcontents = os.listdir(os.path.join(root, directory))
                for file in workbookcontents:
                    filepath = os.path.join(root, directory, file)
                    if os.path.isfile(filepath) and file.endswith('.txt'):
                        with open(filepath) as f:
                            content = f.read()

                        content = content.decode('utf-8')
                        content = re.sub(r'\n', ' ', content)
                        content = re.sub(r'\t', ' ', content)
                        content = re.sub(r' +', ' ', content)
                        assessment.append(content)
                        assessmentname.append(os.path.splitext(os.path.basename(filepath))[0])

    return assessment, assessmentname