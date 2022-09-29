"""
Perform K means clustering on clusters of textbook items
This runs the iterative k-means.
In-order to run the non-iterative k-means, we have to remove the while loop at line number 21 and check for the indentation properly.

"""


from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keyword_extraction import *
from gensim import models


def clustering_docs(workbook, assessment, num_clusters, create_cluster_file, strategy, clustering, encoding, compound):
    new_num_clusters = 0
    flag = False
    while new_num_clusters != num_clusters:
	
        # To check if this is the first loop
    	if flag:
    	    num_clusters = new_num_clusters
        if encoding == '1':
            tfidf_vectorizer = TfidfVectorizer(use_idf=False, stop_words = 'english')
        elif encoding == '2':
	    tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words = 'english')

		
        # TF of workbook

        if strategy == '1':
	    if encoding == '3':
	        if clustering == '2':
	            doc2vecSentence = [models.doc2vec.LabeledSentence(words=assessment[x], tags=["DOC_"+str(x)]) for x in range(0,len(assessment))]
            	    model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
            	    model.build_vocab(doc2vecSentence)
	    	    tfidf_matrix = list(model.docvecs.doctag_syn0)
		    workbook_tdidf = cosine_similarity(tfidf_matrix, tfidf_matrix)
	        if clustering == '1':
		    doc2vecSentence = [models.doc2vec.LabeledSentence(words=assessment[x], tags=["DOC_"+str(x)]) for x in range(0,len(assessment))]
            	    model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
            	    model.build_vocab(doc2vecSentence)
	    	    tfidf_matrix = list(model.docvecs.doctag_syn0)
	    	    workbook_tdidf = tfidf_matrix
	        if clustering == '3':
	    	    coursecontent = assessment + workbook
		    doc2vecSentence = [models.doc2vec.LabeledSentence(words=coursecontent[x], tags=["DOC_"+str(x)]) for x in range(0,len(coursecontent))]
            	    model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
            	    model.build_vocab(doc2vecSentence)
	    	    tfidf_matrix = list(model.docvecs.doctag_syn0)
	    	    cluster_tfidf = tfidf_matrix[0:len(assessment)]
	            assessment_tfidf = tfidf_matrix[len(assessment):]
   	            workbook_tdidf = cosine_similarity(cluster_tfidf, assessment_tfidf)

    	    else:
	        if clustering == '2':
       	            tfidf_matrix = tfidf_vectorizer.fit_transform(assessment)
       	            workbook_tdidf = cosine_similarity(tfidf_matrix, tfidf_matrix)
	        elif clustering == '1':
                    workbook_tdidf = tfidf_vectorizer.fit_transform(assessment)
	        elif clustering == '3':
                    coursecontent = assessment + workbook
    	            tfidf_matrix = tfidf_vectorizer.fit_transform(coursecontent)
 	    	    cluster_tfidf = tfidf_matrix[0:len(assessment)]
    	            assessment_tfidf = tfidf_matrix[len(assessment):]
   	    	    workbook_tdidf = cosine_similarity(cluster_tfidf, assessment_tfidf)

        if strategy == '2':
            if encoding == '3':
    	        if clustering == '2':
	            doc2vecSentence = [models.doc2vec.LabeledSentence(words=workbook[x], tags=["DOC_"+str(x)]) for x in range(0,len(workbook))]
       	            model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
       	            model.build_vocab(doc2vecSentence)
	    	    tfidf_matrix = list(model.docvecs.doctag_syn0)
	       	    workbook_tdidf = cosine_similarity(tfidf_matrix, tfidf_matrix)
	        if clustering == '1':
		    doc2vecSentence = [models.doc2vec.LabeledSentence(words=workbook[x], tags=["DOC_"+str(x)]) for x in range(0,len(workbook))]
                    model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
                    model.build_vocab(doc2vecSentence)
	            tfidf_matrix = list(model.docvecs.doctag_syn0)
	    	    workbook_tdidf = tfidf_matrix
	        if clustering == '3':
	    	    coursecontent = workbook + assessment
		    doc2vecSentence = [models.doc2vec.LabeledSentence(words=coursecontent[x], tags=["DOC_"+str(x)]) for x in range(0,len(coursecontent))]
            	    model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
            	    model.build_vocab(doc2vecSentence)
	    	    tfidf_matrix = list(model.docvecs.doctag_syn0)
	    	    cluster_tfidf = tfidf_matrix[0:len(workbook)]
	            assessment_tfidf = tfidf_matrix[len(workbook):]
   	            workbook_tdidf = cosine_similarity(cluster_tfidf, assessment_tfidf)

	    else:
	        if clustering == '2':
                    tfidf_matrix = tfidf_vectorizer.fit_transform(workbook)
    	            workbook_tdidf = cosine_similarity(tfidf_matrix, tfidf_matrix)
	        elif clustering == '1':
	            workbook_tdidf = tfidf_vectorizer.fit_transform(workbook)
	        elif clustering == '3':
	            coursecontent = workbook + assessment
	            tfidf_matrix = tfidf_vectorizer.fit_transform(coursecontent)
    	            cluster_tfidf = tfidf_matrix[0:len(workbook)]
    	            assessment_tfidf = tfidf_matrix[len(workbook):]
   	            workbook_tdidf = cosine_similarity(cluster_tfidf, assessment_tfidf)


        # Performing K means clustering
        print "\nCreating " + str(num_clusters) + " clusters for you now...."
        print "Please wait...."
        km = KMeans(n_clusters = num_clusters, max_iter = 50000, init = 'k-means++')
        clusterofdocs = defaultdict(lambda: " ")

        X = km.fit_transform(workbook_tdidf)

        # Creating clusters of text of docs (workbooks)
        if strategy == '1':
            for i in range(X.shape[0]):
   	        minindex = np.argmin(X[i])
       	        clusterofdocs[minindex] += " " + assessment[i]
        else:
	    for i in range(X.shape[0]):
                minindex = np.argmin(X[i])
       	        clusterofdocs[minindex] += " " + workbook[i]


        clusters = []

        # creating a list of clusters and cluster names
        for index, cluster in enumerate(clusterofdocs):
            clusters.append(clusterofdocs[cluster])

        # Keyphrase extraction
        new_num_clusters, clusterkeywords = extract_keywords(clusters, compound)

        # Write clusters and their names into a file
        clusternames = []

        for cluster in clusterkeywords:
            clusternames.append(cluster[0])

        flag = True

    if create_cluster_file:
        for index, cluster in enumerate(clusters):
            with open('Results/clusters/clusters.txt', 'a') as f:
                f.write(str(index+1)+'. '+clusternames[index])
                f.write("\n")
                f.write(clusters[index].encode('utf-8'))
                f.write("\n\n")
    return clusters, clusternames
