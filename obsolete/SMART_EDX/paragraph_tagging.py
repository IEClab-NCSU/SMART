from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_similarity(clusters, paragraphs_to_skills, questions_to_skills,
                    skillnames, teks_paragraphs, teks_names):

    content = clusters + teks_paragraphs

    tfidf_vectorizer = TfidfVectorizer(use_idf=False, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(content)

    # To calculate cosine similarity between each cluster and each assessment
    cluster_tfidf = tfidf_matrix[0:len(clusters)]
    teks_paragraphs_tfidf = tfidf_matrix[len(clusters):]

    final_matrix = cosine_similarity(cluster_tfidf, teks_paragraphs_tfidf)

    result = []
    for question in questions_to_skills:
        for i in range(len(skillnames)):
            if skillnames[i] == question[1]:
                result.append([question[0],
                               question[2],
                               teks_names[np.argmax(final_matrix[i, :])],
                               teks_paragraphs[np.argmax(final_matrix[i, :])]])

    result2 = []
    for paragraph in paragraphs_to_skills:
        for i in range(len(skillnames)):
            if skillnames[i] == paragraph[1]:
                result2.append([paragraph[0],
                                paragraph[2],
                                teks_names[np.argmax(final_matrix[i, :])],
                                teks_paragraphs[np.argmax(final_matrix[i, :])]])

    return result, result2
