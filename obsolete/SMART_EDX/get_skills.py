"""
Create skills
Create mapping between paragraphs and skills
Create mapping between questions and skill
"""


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kmeans_clustering import find_skills


# Create Mappings between skills and questions
def skill_question_mapping(skills, questions):

    # Create a list of text of skills
    skills_text = []
    for skill in skills:
        skills_text.append(skill[1])

    # Create a list of text of questions
    questions_text = []
    for question in questions:
        questions_text.append(question[1])

    skills_questions = questions_text + skills_text

    # Set parameters for calculating TF
    tfidf_vectorizer = TfidfVectorizer(use_idf=False, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(skills_questions)

    # To calculate cosine similarity between skills and questions
    cluster_tfidf = tfidf_matrix[0:len(skills)]
    assessment_tfidf = tfidf_matrix[len(skills):]

    # Computing Cosine Similarity
    final_matrix = cosine_similarity(cluster_tfidf, assessment_tfidf)

    questions_to_skills = []

    for i in range(len(questions)):
        questions_to_skills.append([questions[i][0], skills[np.argmax(final_matrix[:, i])][0], questions[i][1]])

    return questions_to_skills


def create_mapping(paragraphs, questions, num_clusters, iterative_clustering):

    # Create skills and map skills to paragraphs
    skills, questions_to_skills, clusters, skillnames = find_skills(questions, num_clusters, iterative_clustering)

    # Create mappings between skills and questions
    paragraphs_to_skills = skill_question_mapping(skills, paragraphs)

    return paragraphs_to_skills, clusters, questions_to_skills, skillnames
