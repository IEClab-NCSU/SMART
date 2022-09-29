from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import csv

# create a list of workbook contents
def create_workbook_lists():
    workbook = []
    workbookname=[]
    for root, directories, filenames in os.walk('content/'):
        for directory in directories:
            if directory == "x-oli-workbook_page":
                workbookcontents = os.listdir(os.path.join(root, directory))
                for file in workbookcontents:
                    filepath = os.path.join(root, directory, file)
                    if os.path.isfile(filepath) and file.endswith('.txt'):
                        with open(filepath) as f:
                            content = f.read()
                        workbook.append(content)
                        workbookname.append(os.path.basename(filepath))
    return workbook, workbookname


# create a list of assessment items
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
                        assessment.append(content)
                        assessmentname.append(os.path.splitext(os.path.basename(filepath))[0])

    return assessment, assessmentname



# Create Associations between workbook text and assessment items

def workbook_assessment_matches():

    assessment, assessmentname = create_assessments_list()
    workbook, workbookname = create_workbook_lists()

    coursecontent = workbook + assessment

    tfidf_vectorizer = TfidfVectorizer(use_idf=False, sublinear_tf=True)

    tfidf_matrix = tfidf_vectorizer.fit_transform(coursecontent)


    workbook_tfidf = tfidf_matrix[0:len(workbook)]
    assessment_tfidf = tfidf_matrix[len(workbook): ]


    final_matrix = cosine_similarity(workbook_tfidf, assessment_tfidf)
    result = []
    for i in range(len(assessment)):
        result.append([assessmentname[i], workbookname[np.argmax(final_matrix[:, i])]])

    final_matrix = np.asarray(final_matrix)
    np.savetxt("Results/workbook_assessment_similarity.csv", final_matrix, delimiter=",")

    with open('Results/workbook_assessment_matches.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(result)

    print "Created Associations between individual questions and paragraphs..."




#workbook_assessment_matches()