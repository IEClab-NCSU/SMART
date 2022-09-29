"""
Create a Mapping of Assessments and Skills

"""

import csv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kmeansClustering import clustering_docs


# Create Associations between cluster of workbook text and assessment items
def cluster_assessment_mapping(num_clusters, num_iterations, workbook, assessment):

    # Creating Clusters...
    print "Now, I will create optimal clusters for you..."
    clusters, clusternames = clustering_docs(workbook, num_clusters, num_iterations)

    coursecontent = clusters + assessment

    tfidf_vectorizer = TfidfVectorizer(use_idf = False, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(coursecontent)

    # To calculate cosine similarity between each cluster and each assessment
    cluster_tfidf = tfidf_matrix[0:len(clusters)]
    assessment_tfidf = tfidf_matrix[len(clusters):]

    print "Creating mappings between assessment items and skills now..."

    # Computing Cosine Similarity
    final_matrix = cosine_similarity(cluster_tfidf, assessment_tfidf)
    #print len(assessment)
    # Creating a list of mappings between assessment ---> cluster name
    result = []
    for i in range(len(assessment)):
        result.append([assessment[i], clusternames[np.argmax(final_matrix[:, i])]])

    # Saving the csv file containing 2 columns.. 1. Assessment 2. Cluster name
    dirname = os.path.split(os.path.abspath(__file__))
    filename = dirname[0] +'/Results/assessment_skill.csv'
    #print len(result)
    print "Created ", filename
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(result)

    return filename, clusternames
