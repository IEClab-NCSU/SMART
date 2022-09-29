"""
Cluster and identify the skill names for text1 (assessments or paragraphs)
"""
from collections import Counter
import csv
import os

from .clustering import doClustering
from .utils.safe_open import safe_open_w    

def text1_skill_mapping(text_ids1, lemmatized_texts1, original_texts1, parameters, is_called_from_service = False):
	strategyType, encodingType, clusteringType, clusters, save_clusterKeywordMapping, save_assessmentSkillMapping, save_paragraphSkillMapping, inputFolder, outputFolder, n_run, course, merge_status  = parameters
	
	if clusters == None:
		kmeansType = 'iterative'
		# current_num_clusters = len(set(lemmatized_texts1))
		current_num_clusters = int(len(lemmatized_texts1)/2)
	else:
		kmeansType = 'non-iterative'
		current_num_clusters = int(clusters)
	
	new_num_clusters = current_num_clusters
	isFirstIteration = True
	while isFirstIteration or new_num_clusters != current_num_clusters:
		print('Number of Clusters:', current_num_clusters, '\n')
		isFirstIteration = False
		current_num_clusters = new_num_clusters

		# print('Performing clustering for k = ',current_num_clusters) # To update
		# for text in lemmatized_texts1:
			# print(text, '\n')
		text_id1_to_clusterIndex, clusteredText1_to_skill, clusterIndex_to_skill, new_num_clusters = doClustering(encodingType, clusteringType, current_num_clusters, text_ids1, lemmatized_texts1, original_texts1, kmeansType, merge_status)
		if kmeansType == 'non-iterative':
			break	
	#end of while loop
	
	# Creating a list of mappings between text_ids1 ---> skill name
	# using transitive mapping: text_ids1 -> clusterIndex -> skill
	text_id1_to_skill = dict()
	for text_id1 in text_ids1:
		clusterIndex = text_id1_to_clusterIndex[text_id1]
		text_id1_to_skill[text_id1] = clusterIndex_to_skill[clusterIndex]

	# modified 5/26/2021: combine clusteredText if skill names match
	skill_to_clusteredText1 = {}
	for clusteredText, skill in clusteredText1_to_skill.items():
		if skill in skill_to_clusteredText1:
			skill_to_clusteredText1[skill] += clusteredText
		else:
			skill_to_clusteredText1[skill] = clusteredText

	new_clusteredText1_to_skill = {}
	for skill, clusteredText in skill_to_clusteredText1.items():
		new_clusteredText1_to_skill[clusteredText] = skill
	# end of updated code

	#Defining output file name
	parameter_details = course + '_' + clusteringType + '_' + encodingType + '_' + str(new_num_clusters) + '_' + strategyType
	
	# Write clusters and their names into a .txt file
	if save_clusterKeywordMapping:
		#file = os.path.join(dataFolder, outputFolderName, strategy + 'Clusters_skill_' + str(new_num_clusters) + '.txt')
		file = os.path.join(outputFolder, n_run + '_' + strategyType + 'Clusters_skill_' + str(new_num_clusters) + '_' + parameter_details + '.csv')
		
		with safe_open_w(file) as f: #safe_open_w() opens file in w mode by creating parent directories if needed
		#with open(file, 'w') as f:
			writer = csv.writer(f, delimiter = ',')
			index = 1
			for clusteredText, skill in new_clusteredText1_to_skill.items():
				row = [str(index) + '. ', skill, clusteredText]
				writer.writerow(row)
				index += 1

	# Write text_ids1 and their cluster names into a file
	save_text1SkillMapping = save_assessmentSkillMapping if strategyType == 'assessment' else save_paragraphSkillMapping
	if save_text1SkillMapping:
		file = os.path.join(outputFolder, n_run + '_' + strategyType + '_skill_' + parameter_details + '.csv')
		with safe_open_w(file) as fin:
			writer = csv.writer(fin, delimiter = ',')
			for text_ids1, skill in text_id1_to_skill.items():
				writer.writerow([text_ids1, skill])
		#print "Created ", file

	if is_called_from_service: # if called by SMART_CyberBook, return the skill names for the text ids
		return text_id1_to_skill

	return new_clusteredText1_to_skill
