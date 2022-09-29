"""
Perform K means Clustering on the Text Items

"""


from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from keyword_extraction import *
import csv
import os


def clustering_docs(workbook, num_clusters, num_iterations):
    new_num_clusters = 0
    flag = False
    while new_num_clusters != num_clusters:

        # To check if this is the first loop
        if flag:
            num_clusters = new_num_clusters
        tfidf_vectorizer = TfidfVectorizer(use_idf=False, stop_words = 'english')

        # TF of workbook
        workbook_tdidf = tfidf_vectorizer.fit_transform(workbook)

        # Performing K means clustering
        print "\nCreating " + str(num_clusters) + " clusters for you now...."
        print "Please wait...."
        km = KMeans(n_clusters = num_clusters, max_iter = num_iterations, init = 'k-means++')

        clusterofdocs = defaultdict(lambda: " ")

        X = km.fit_transform(workbook_tdidf)
        text_skill_mapping = []
        # Creating clusters of text of docs (workbooks)
        for i in range(X.shape[0]):
            minindex = np.argmin(X[i])
            clusterofdocs[minindex] += " " + workbook[i]

        clusters = []

        # creating a list of clusters and cluster names
        for index, cluster in enumerate(clusterofdocs):
            clusters.append(clusterofdocs[cluster])

        # Keyphrase extraction
        new_num_clusters, clusterkeywords = extract_keywords(clusters)

        # Write clusters and their names into a file
        clusternames = []

        for cluster in clusterkeywords:
            clusternames.append(cluster[0])

        text_skill_mapping = []
        # Creating clusters of text of docs (workbooks)
        for i in range(X.shape[0]):
            minindex = np.argmin(X[i])
            text_skill_mapping.append([workbook[i].encode('utf-8'), clusternames[minindex]])

        flag = True
    dirname = os.path.split(os.path.abspath(__file__))

    with open(dirname[0]+'/Results/text_skill.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(text_skill_mapping)


    return clusters, clusternames
