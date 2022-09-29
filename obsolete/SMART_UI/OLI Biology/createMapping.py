import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kmeansClustering import clustering_docs
from check_OLIdatavsDatashop_with_subsets import correctstepname
from create_content_lists import *


# Create Associations between cluster of workbook text and assessment items
def cluster_assessment_mapping(use_sentence, num_clusters, create_file, create_cluster_file, create_clusters):

    # Creating a list of all assessment items and their names
    assessment, assessmentname = create_assessments_list()
    print "The number of assessments is .. ", len(assessment)

    # Creating a list of all workbook items and their names
    workbook, workbookname = create_workbook_lists(use_sentence)

    print "The number of workbook items is... ", len(workbook)

    # Creating Clusters...
    if create_clusters:
        print "Now, I will create optimal clusters for you..."
        clusters, clusternames = clustering_docs(workbook, num_clusters, create_cluster_file)
    else:
        clusters = workbook
        clusternames = workbookname

    coursecontent = clusters + assessment

    tfidf_vectorizer = TfidfVectorizer(use_idf = False, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(coursecontent)

    # To calculate cosine similarity between each cluster and each assessment
    cluster_tfidf = tfidf_matrix[0:len(clusters)]
    assessment_tfidf = tfidf_matrix[len(clusters):]
    print "Creating mappings between assessment items and skills now..."

    # Computing Cosine Similarity
    final_matrix = cosine_similarity(cluster_tfidf, assessment_tfidf)

    # Creating a list of mappings between assessment ---> cluster name
    result = []
    for i in range(len(assessment)):
        result.append([assessmentname[i], clusternames[np.argmax(final_matrix[:, i])]])

    if create_file:
        new_assessmentname = []
        with open('Corrected Discrepancies in OLI Data/corrected_cluster_assessment_matches_2_columns.csv',
                  'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                new_assessmentname.append(row[0])

        finalresult = []

        for index, r in enumerate(result):
            finalresult.append([new_assessmentname[index], r[1]])
        for i in range(100):
            if use_sentence and \
                            os.path.isfile('Results/Sentence_' + str(
                                len(clusters)) + 'cluster_assessment_matches' + '_v' + str(
                                i) + '.csv') == False:
                filename = 'Results/Sentence_' + str(
                    len(clusters)) + 'cluster_assessment_matches' + '_v' + str(i) + '.csv'
                break
            elif not use_sentence and \
                            os.path.isfile('Results/Paragraph_' + str(
                                len(clusters)) + 'cluster_assessment_matches' + '_v' + str(
                                i) + '.csv') == False:
                filename = 'Results/Paragraph_' + str(
                    len(clusters)) + 'cluster_assessment_matches' + '_v' + str(i) + '.csv'
                break

        print "Created ", filename
        with open(filename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(finalresult)

        correctstepname(filename)
