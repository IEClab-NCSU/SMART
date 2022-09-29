"""
Download nltk and the punkt package and inflection (Check the main ReadMe file for the commands).

In-order to check if the download is successful, run the following commands on the console -
python
import nltk

If this does not show any error then nltk is successfully installed.

If the download is successful, then uncomment line numbers 18, 19, 20, 21, 40, 41, 42, 43, 44, 68, 69, 70, 71.
Change line number 45 and 72 to workbook.append(res) and assessment.append(res) respectively.
This is done as a part of preprocessing where we convert all the words in our text to singular using the inflection library supported by python

"""

import os
import re
#import inflection
#from nltk import sent_tokenize
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize

# create a list of workbook contents
def create_workbook_lists():
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
	    content = re.sub(r'[^\w\d\s]', ' ', content)
	    #text_tokens = word_tokenize(content)
	    #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
	    #res = ' '
	    #for i in tokens_without_sw:
	    	#res += inflection.singularize(i) + ' '
            workbook.append(content)
            workbookname.append(os.path.basename(filepath))

    return workbook, workbookname


# Create a list of assessment items
def create_assessments_list():
    assessment = []
    assessmentname = []
    directory = 'Assessments/'
    assessmentcontents = os.listdir(directory)
    for file in assessmentcontents:
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath) and file.endswith('.txt'):
            with open(filepath) as f:
                content = f.read()

            content = content.decode('utf-8')
            content = re.sub(r'\n', ' ', content)
            content = re.sub(r'\t', ' ', content)
            content = re.sub(r' +', ' ', content)
	    content = re.sub(r'[^\w\d\s]', ' ', content)
	    #text_tokens = word_tokenize(content)
	    #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
	    #res = ' '
	    #for i in tokens_without_sw:
	    	#res += inflection.singularize(i) + ' '

            assessment.append(content)
            assessmentname.append(os.path.basename(filepath))

    return assessment, assessmentname