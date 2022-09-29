"""
Main Function and Preprocessing

"""
from get_Data_from_SQL import get_content
from preprocessing import preprocess
from get_skills import create_mapping
from paragraph_tagging import find_similarity
from insert_standard_references import insert_TEKS
from insert_skills import insert_skills
from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=['POST'])
def main():

    paragraphs, questions = get_content()

    paragraphs = preprocess(paragraphs)
    questions = preprocess(questions)

    num_clusters = 8
    # Find Clusters
    paragraphs_to_skills, clusters, questions_to_skills, skillnames = \
        create_mapping(paragraphs, questions, num_clusters,
                       iterative_clustering=False)

    # Get Standard IDs and descriptions
    data = request.json
    standardID = data['standardID']
    standardDescription = data['standardDescription']

    # Find Cosine Similarity between Paragraphs and TEKS
    questions_to_standard, paragraphs_to_standard = find_similarity(clusters, paragraphs_to_skills, questions_to_skills,
                                             skillnames, standardDescription, standardID)

    insert_skills(paragraphs_to_skills, questions_to_skills)

    insert_TEKS(paragraphs_to_standard, questions_to_standard)

    return ""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
