from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keyword_extraction import *
from gensim import models
from collections import defaultdict
import sys

def doClustering(encodingType, clusteringType, compoundWords, current_num_clusters, text1, text1name):
	if encodingType == 'w2v': #w2v encodingType (to be verified)
		if clusteringType == 'first': #first level
			doc2vecSentence = [models.doc2vec.LabeledSentence(words=text1[x], tags=["DOC_"+str(x)]) for x in range(0,len(text1))]
			model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
			model.build_vocab(doc2vecSentence)
			tfidf_matrix = list(model.docvecs.doctag_syn0)
			vectors = tfidf_matrix
		elif clusteringType == 'second': #second level
			doc2vecSentence = [models.doc2vec.LabeledSentence(words=text1[x], tags=["DOC_"+str(x)]) for x in range(0,len(text1))]
			model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
			model.build_vocab(doc2vecSentence)
			tfidf_matrix = list(model.docvecs.doctag_syn0)
			vectors = cosine_similarity(tfidf_matrix, tfidf_matrix)
		
		elif clusteringType == 'hybrid': #hybrid
			coursecontent = text1 + workbook
			doc2vecSentence = [models.doc2vec.LabeledSentence(words=coursecontent[x], tags=["DOC_"+str(x)]) for x in range(0,len(coursecontent))]
			model = models.Doc2Vec(dbow_words =1, min_count = 50, epochs = 40 ,)
			model.build_vocab(doc2vecSentence)
			tfidf_matrix = list(model.docvecs.doctag_syn0)
			cluster_tfidf = tfidf_matrix[0:len(text1)]
			text1_tfidf = tfidf_matrix[len(text1):]
			vectors = cosine_similarity(cluster_tfidf, text1_tfidf)
		else:
			print "Invalid input for clustering type."
			sys.exit()
	else:
		if encodingType == 'tf': #tf encodingType
			tfidf_vectorizer = TfidfVectorizer(use_idf=False, stop_words = 'english')
		elif encodingType == 'tfidf': #tfidf encodingType
			tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words = 'english')
		else:
			print "Invalid input for encoding type."
			sys.exit()

		if clusteringType == 'first': #first level
			vectors = tfidf_vectorizer.fit_transform(text1)
		elif clusteringType == 'second': #second level
			tfidfs = tfidf_vectorizer.fit_transform(text1)
			vectors = cosine_similarity(tfidfs, tfidfs)
		elif clusteringType == 'hybrid': # hybrid: to be corrected
			coursecontent = text1 + text2
			tfidf_matrix = tfidf_vectorizer.fit_transform(coursecontent)
			text1_cluster_tfidf = tfidf_matrix[0:len(text1)] #incomplete. text1_cluster_tfidf does not store tfidf of 'the clusters' of text 1
			text2_tfidf = tfidf_matrix[len(text1):]
			vectors = cosine_similarity(text1_cluster_tfidf, text2_tfidf)
		else:
			print "Invalid input for clustering type."
			sys.exit()
		#To improve: I think, text 1 is not changing. Why vectorize the same thing redundantly?

	# len(vectors) is equal to len(text1) for first and second level clusteringType.

	# Performing K means clusteringType
	#print "\nCreating " + str(current_num_clusters) + " clusters for you now...."
	#print "Please wait...."
	km = KMeans(n_clusters = current_num_clusters, max_iter = 50000, init = 'k-means++')

	X = km.fit_transform(vectors) 
	# X is a matrix representing cluster-distance space

	# Creating clusters of text1
	clusterofdocs = defaultdict(lambda: " ")
	text1_clusterIndex_mapping = dict()
	for i in range(X.shape[0]):
		minIndex = np.argmin(X[i])
		clusterofdocs[minIndex] += " " + text1[i]

		text1_clusterIndex_mapping[text1name[i]] = minIndex

	no_of_clusters = len(clusterofdocs)

	# creating a list of clusters. Order of clusters' items is important here.
	clusters = [None]*no_of_clusters
	for clusterIndex in clusterofdocs.keys():
		clusters[clusterIndex] = clusterofdocs[clusterIndex]

	# Keyphrase extraction
	new_num_clusters, clusterkeywords = extract_keywords(clusters, compoundWords)

	clusternames = []
	for cluster in clusterkeywords:
		clusternames.append(cluster[0]) 
	#Investigate(above): Why do we change clusterkeywords to clusternames. Why only first word?
	return clusternames, text1_clusterIndex_mapping, clusters, new_num_clusters
