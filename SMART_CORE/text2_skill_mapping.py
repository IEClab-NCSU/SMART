"""
Input: hyperparameters and configurations (to save chosen output files)
Output: Generate mappings and save on disk 
"""
#!/usr/bin/env python3
import os
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import sys

# from average_embedding import get_average_embedding
from .fineTuned_embeddings import get_fineTuned_embeddings, get_fineTuned_embeddings_updated, get_embeddings_pretrainBERT, get_embeddings_custom_pretrainBERT, get_embeddings_sentBert
from .text1_skill_mapping import text1_skill_mapping
from .create_input_lists import create_input_lists_from_csv, lemmatize
#from gensim import models

def text2_skill_mapping(parameters):

	strategyType, encodingType, clusteringType, clusters, save_clusterKeywordMapping, save_assessmentSkillMapping, save_paragraphSkillMapping, inputFolder, outputFolder, n_run, course, merge_status = parameters


	#Inputs as csv files
	input1 = 'assessments.csv' if strategyType == 'assessment' else 'paragraphs.csv'
	input2 = 'paragraphs.csv' if strategyType == 'assessment' else 'assessments.csv'

	input1_path = os.path.join(inputFolder, input1)
	input2_path = os.path.join(inputFolder, input2)

	"""
	Fetch preprocessed and original texts from the csv files.
		1. The preprocessed texts for generating embeddings.
		2. The original texts would be later used for text rank since it requires original text (including stopwords) for combining keywords.
	"""
	text_ids1, lemmatized_texts1, original_texts1, text_ids2, lemmatized_texts2, original_texts2 = create_input_lists_from_csv(input1_path, input2_path)
	
	##############
	#compute the average of the pairwise similarity of sentences among paragraphs
	# from paragraph_sentences import interParagraphSimilarity
	# interParagraphSimilarity(lemmatized_text1, strategyType)
	##############

	clusteredText1_to_skill = text1_skill_mapping(text_ids1, lemmatized_texts1, original_texts1, parameters)
		
	clusteredLemmatizedTexts1 = []
	clusteredLemmatizedText1_skills = []
	for text, skill in clusteredText1_to_skill.items():
		clusteredLemmatizedTexts1.append(lemmatize(text))
		clusteredLemmatizedText1_skills.append(skill)

	if encodingType == 's-bert':
		clusteredLemmatizedText1_lemmatizedText2_embeddings = get_embeddings_sentBert(clusteredLemmatizedTexts1 + lemmatized_texts2)
	elif encodingType == 'bert':
		clusteredLemmatizedText1_lemmatizedText2_embeddings = get_embeddings_pretrainBERT(clusteredLemmatizedTexts1 + lemmatized_texts2)
	else:
		if encodingType == 'tf':
			vectorizer = TfidfVectorizer(use_idf = False, stop_words='english')
		elif encodingType == 'tfidf':
			vectorizer = TfidfVectorizer(use_idf=True, stop_words = 'english')
		else:
			print("Invalid input for encoding type.")
			sys.exit()

		clusteredLemmatizedText1_lemmatizedText2_embeddings = vectorizer.fit_transform(clusteredLemmatizedTexts1 + lemmatized_texts2)
		
		# pca = PCA(n_components = 50)
		# clusteredLemmatizedText1_lemmatizedText2_embeddings = pca.fit_transform(clusteredLemmatizedText1_lemmatizedText2_embeddings.todense())

	# Compute cosine similarity between each clusteredLemmatizedText1 and text2 (Section 3.4 Skill Mapping)
	clusteredLemmatizedText1_embeddings = clusteredLemmatizedText1_lemmatizedText2_embeddings[0:len(clusteredLemmatizedTexts1)]
	lemmatizedText2_embeddings = clusteredLemmatizedText1_lemmatizedText2_embeddings[len(clusteredLemmatizedTexts1):]

	similarity_matrix = cosine_similarity(clusteredLemmatizedText1_embeddings, lemmatizedText2_embeddings)
	
	# Creating a list of mappings between text2 ---> skill name
	text2_to_skill = []
	for i in range(len(lemmatized_texts2)):
		skill = clusteredLemmatizedText1_skills[np.argmax(similarity_matrix[:, i])]
		text2_to_skill.append([text_ids2[i], skill])

	save_text2SkillMapping = save_paragraphSkillMapping if strategyType == 'assessment' else save_assessmentSkillMapping
	if save_text2SkillMapping:
		other_strategy = 'paragraph_' if strategyType == 'assessment' else 'assessment_'
		parameter_details = course + '_' + clusteringType + '_' + encodingType + '_' + str(len(clusteredLemmatizedText1_skills)) + '_' + strategyType
		
		# Write text_ids2 and their cluster names into a file
		file = os.path.join(outputFolder, n_run + '_' + other_strategy + 'skill_' + parameter_details + '.csv')
		with open(file, 'w') as f:
			writer = csv.writer(f)
			writer.writerows(text2_to_skill)
		#print "Created ", file

		print(outputFolder + '/' + n_run + '_' + 'assessment_skill_' + parameter_details + '.csv')
	#End of SMART-CORE
