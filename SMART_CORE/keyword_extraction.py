"""
Extract keywords using the textrank algorithm.
Note: We modified the implementation of the textrank algorithm in the summa package
"""

import re
from collections import Counter

"""
Using local (modified) summa package
"""
from .summa.keywords import keywords
from .custom_stopwords import get_custom_stopwords
from .create_input_lists import lemmatize

def extract_keywords(clusterIndex_to_clusteredText1, kmeansType, merge_status):
	clusteredText1_to_skill = dict()
	clusterIndex_to_skill = dict()

	# add 7/28/2021 for tracking count of skill names
	skill_to_count = {}
	for clusterIndex, clusteredText in clusterIndex_to_clusteredText1.items():
		if not clusteredText.encode('utf-8'):
			skill_name = 'UNKNOWN'
		else:
			skill_names = keywords(clusteredText, additional_stopwords = get_custom_stopwords()) #by default: ratio (default = 0.2)

		top_skill_name = skill_names.split('\n')[0]
		if len(top_skill_name) == 0:
			# modified on 5/19/2021: changed to select most frequent lemma instead of most frequent original word
			words_freq = Counter(lemmatize(clusteredText).split())
			maxFreq = 0
			for word, freq in words_freq.items():
				if freq > maxFreq:
					top_skill_name = word
					maxFreq = freq
			
		if not top_skill_name:
			top_skill_name = 'UNKNOWN'

		if kmeansType == 'non-iterative' and merge_status == 'off':
			if top_skill_name in skill_to_count:
				skill_to_count[top_skill_name] += 1
				top_skill_name = top_skill_name + '_' + str(skill_to_count[top_skill_name])
			else:
				skill_to_count[top_skill_name] = 1

		clusteredText1_to_skill[clusteredText] = top_skill_name
		clusterIndex_to_skill[clusterIndex] = top_skill_name
		#print('....')
		#print(skill_name)
		
	distinct_skill_count = len(set(clusterIndex_to_skill.values())) # new_num_clusters

	return clusteredText1_to_skill, clusterIndex_to_skill, distinct_skill_count
