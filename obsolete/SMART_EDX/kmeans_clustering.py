"""
Perform K means clustering to get skills from paragraphs
"""

from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from keyword_extraction_with_Watson import *


def find_skills(paragraphs, num_clusters, iterative_clustering):

    paragraphs_text = []

    paragraphs_dict = defaultdict(lambda: [])

    for paragraph in paragraphs:

        if paragraph[1] not in paragraphs_dict:
            paragraphs_text.append(paragraph[1])
            paragraphs_dict[paragraph[1]].append(paragraph[0])

    new_num_clusters = 0
    flag = False

    while new_num_clusters != num_clusters:

        # To check if this is the first loop
        if flag:
            num_clusters = new_num_clusters

        tfidf_vectorizer = TfidfVectorizer(use_idf=False, stop_words='english')

        # Term Frequency of paragraphs
        paragraphs_tfidf = tfidf_vectorizer.fit_transform(paragraphs_text)

        # Define K Means Clustering parameters
        km = KMeans(n_clusters=num_clusters, max_iter=100000, init='k-means++')

        clusterofdocs = defaultdict(lambda: " ")

        X = km.fit_transform(paragraphs_tfidf)

        # Creating clusters of paragraphs
        for i in range(X.shape[0]):
            minindex = np.argmin(X[i])
            paragraphs_dict[paragraphs_text[i]].append(minindex)
            clusterofdocs[minindex] += " " + paragraphs_text[i]

        clusters = [""] * num_clusters

        # creating a list of clusters and cluster names
        for index, cluster in enumerate(clusterofdocs):
            clusters[cluster] = clusterofdocs[cluster]

        # Skill Name extraction
        new_num_clusters, skillnames = get_skillnames(clusters, iterative_clustering)

        """
        Create a list of tuples with 
        first element being the skill name
        and the second one being 
        the content of that skill
        """
        skills_content = []
        for index, cluster in enumerate(clusters):
            skills_content.append((skillnames[index], clusters[index]))

        for text in paragraphs_dict:
            paragraphs_dict[text][-1] = skillnames[int(paragraphs_dict[text][-1])]

        """
        Convert dictionary of paragraphs to a list 
        with first element being the xblock id, 
        second being skill name and third
        being the text of paragraph
        """
        paragraphs_to_skills = []
        for key in paragraphs_dict:
            if len(paragraphs_dict[key]) <= 2:
                paragraphs_to_skills.append([paragraphs_dict[key][0],
                                            paragraphs_dict[key][1],
                                            key])
            else:
                paragraphs_to_skills.append([paragraphs_dict[key][0],
                                            paragraphs_dict[key][2],
                                            key])

                paragraphs_to_skills.append([paragraphs_dict[key][1],
                                            paragraphs_dict[key][2],
                                            key])

        if not iterative_clustering:
            break

        flag = True

    return skills_content, paragraphs_to_skills, clusters, skillnames
