"""Cluster the assessments using K-means and generate skills
"""
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import sys

# from .utils.average_embedding import get_average_embedding
from .fineTuned_embeddings import get_fineTuned_embeddings, get_fineTuned_embeddings_updated, get_embeddings_pretrainBERT, get_embeddings_custom_pretrainBERT, get_embeddings_sentBert
from .keyword_extraction import *

"""
# Embedding analysis
from .analysis.compute_stats import compute_stats
from .analysis.variance_analysis import variance_analysis
from .analysis.compute_interClusterSimilarity import interClusterSimilarity
"""

def doClustering(encodingType, clusteringType, current_num_clusters, text_ids1, lemmatized_texts1, original_texts1, kmeansType, merge_status):
	if encodingType == 's-bert':
		vectors = get_embeddings_sentBert(lemmatized_texts1)
		if clusteringType == 'second':
			vectors = cosine_similarity(vectors, vectors)
		# pca = PCA(n_components = 8)
		# vectors = pca.fit_transform(vectors)
	elif encodingType == 'bert':
		vectors = get_embeddings_pretrainBERT(lemmatized_texts1)
		if clusteringType == 'second':
			vectors = cosine_similarity(vectors, vectors)
		# pca = PCA(n_components = 8)
		# vectors = pca.fit_transform(vectors)
	else:
		if encodingType == 'tf': #tf encodingType
			vectorizer = TfidfVectorizer(use_idf=False, stop_words = 'english')
		elif encodingType == 'tfidf': #tfidf encodingType
			vectorizer = TfidfVectorizer(use_idf=True, stop_words = 'english')
		else:
			print("Invalid input for encoding type.")
			sys.exit()

		vectors = vectorizer.fit_transform(lemmatized_texts1)
		if clusteringType == 'second':
			vectors = cosine_similarity(vectors, vectors)
		else:
			vectors = vectors.todense()

		# pca = PCA(n_components = 30)
		# vectors = pca.fit_transform(vectors)
		
	#####
	# compute_stats(vectors, encodingType, clusteringType) #Computes some stats related to the distance matrix of vectors	
	# sys.exit()
	####

	# Performing K means clusteringType
	#print "\nCreating " + str(current_num_clusters) + " clusters for you now...."
	#print "Please wait...."
	km = KMeans(n_clusters = current_num_clusters, max_iter = 50000, init = 'k-means++')

	#X = km.fit_transform(vectors) 
	# X is a matrix representing cluster-distance space
	cluster_assignment = km.fit(vectors).labels_ #list: ith index holds the value of the corresponding cluster index
	
	"""
	# analysis
	interClusterSimilarity(encodingType, clusteringType, current_num_clusters, cluster_assignment, vectors)
	"""
	# Creating clusters of text1
	clusterIndex_to_clusteredText1 = defaultdict(lambda: " ")	
	text_id1_to_clusterIndex = dict()

	for i, clusterIndex in enumerate(cluster_assignment): # len(cluster_assignment) is equal to len(vectors)
		clusterIndex_to_clusteredText1[clusterIndex] += original_texts1[i] + ". " 
		text_id1_to_clusterIndex[text_ids1[i]] = clusterIndex

	# print(clusterIndex_to_clusteredText1)
	clusteredText1_to_skill, clusterIndex_to_skill, new_num_clusters = extract_keywords(clusterIndex_to_clusteredText1, kmeansType, merge_status)

	#################
	# variance_analysis(vectors, cluster_assignment, clusterIndex_to_skill)
	# sys.exit()
	#################
	
	return text_id1_to_clusterIndex, clusteredText1_to_skill, clusterIndex_to_skill, new_num_clusters
