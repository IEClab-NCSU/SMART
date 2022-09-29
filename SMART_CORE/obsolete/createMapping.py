import csv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kmeansClustering import clustering_docs
from check_OLIdatavsDatashop_with_subsets import correctstepname
from create_content_lists import *
from gensim import models

# Create Associations between cluster of workbook text and assessment items
def cluster_assessment_mapping(num_clusters, create_file, create_cluster_file, create_clusters, strategy, clustering, encoding, compound):

    # Creating a list of all assessment items and their names
    assessment, assessmentname = create_assessments_list()
    print "The number of assessments is .. ", len(assessment)

    # Creating a list of all workbook items and their names
    workbook, workbookname = create_workbook_lists()

    print "The number of workbook items is... ", len(workbook)

    # Creating Clusters...
    if create_clusters:
        print "Now, I will create optimal clusters for you..."
	clusters, clusternames = clustering_docs(workbook, assessment, num_clusters, create_cluster_file, strategy, clustering, encoding, compound)
    else:
	if strategy == '1':
        	clusters = assessment
        	clusternames = assessmentname 
	else:
		clusters = workbook
        	clusternames = workbookname

    if strategy == '1':
    	coursecontent = clusters + workbook
    else:
    	coursecontent = clusters + assessment

    if encoding == '1':
    	tfidf_vectorizer = TfidfVectorizer(use_idf = False, stop_words='english')
	tfidf_matrix = tfidf_vectorizer.fit_transform(coursecontent)
    elif encoding == '2':
    	tfidf_vectorizer = TfidfVectorizer(use_idf = True, stop_words='english')
	tfidf_matrix = tfidf_vectorizer.fit_transform(coursecontent)
    elif encoding == '3': 
        doc2vecSentence = [models.doc2vec.LabeledSentence(words=coursecontent [x], tags=["DOC_"+str(x)]) for x in range(0,len(coursecontent)) ]
        model = models.Doc2Vec(dbow_words = 1, min_count = 50, epochs = 40 ,)
        model.build_vocab(doc2vecSentence)
	tfidf_matrix = list(model.docvecs.doctag_syn0)

    # To calculate cosine similarity between each cluster and each assessment
    cluster_tfidf = tfidf_matrix[0:len(clusters)]
    assessment_tfidf = tfidf_matrix[len(clusters):]

    print "Creating mappings between assessment items and skills now..."

    # Computing Cosine Similarity
    final_matrix = cosine_similarity(cluster_tfidf, assessment_tfidf)
    # Creating a list of mappings between assessment ---> cluster name
    result = []
    for i in range(len(assessment)):
        result.append([os.path.splitext(assessmentname[i])[0], clusternames[np.argmax(final_matrix[:, i])]])


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
	
	if clustering == '1':
		if encoding == '1':
			name = 'First-level_TF_'
		if encoding == '2':
			name = 'First-level_TF-IDF_'
		if encoding == '3':
			name = 'First-level_W2V_'
	elif clustering == '2':
		if encoding == '1':
			name = 'Second-level_TF_'
		if encoding == '2':
			name = 'Second-level_TF-IDF_'
		if encoding == '3':
			name = 'Second-level_W2V_'
	elif clustering == '3':
		if encoding == '1':
			name = 'Hybrid_TF_'
		if encoding == '2':
			name = 'Hybrid_TF-IDF_'
		if encoding == '3':
			name = 'Hybrid_W2V_'

        for i in range(100):
            if strategy == '1' and \
                            os.path.isfile('Results/Assessment_' + name + str(
                                len(clusters)) + 'cluster_assessment_matches' + '_v' + str(
                                i) + '.xlsx') == False:
                filename = 'Results/Assessment_' + name + str(
                    len(clusters)) + 'cluster_assessment_matches' + '_v' + str(i) + '.xlsx'
		fin = 'Assessment_' + name + str(num_clusters) + '_v' + str(i)
                break
            elif strategy == '2' and \
                            os.path.isfile('Results/Paragraph_' + name + str(
                                len(clusters)) + 'cluster_assessment_matches' + '_v' + str(
                                i) + '.xlsx') == False:
                filename = 'Results/Paragraph_' + name + str(
                    len(clusters)) + 'cluster_assessment_matches' + '_v' + str(i) + '.xlsx'
		fin = 'Paragraph_' + name + str(num_clusters) + '_v' + str(i)
                break

        print "Created ", filename
        with open(filename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(finalresult)
        correctstepname(filename, fin)