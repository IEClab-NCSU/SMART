#!/usr/bin/env python3

import argparse
from SMART_CORE.text2_skill_mapping import text2_skill_mapping
import os
import sys


def main():
	print('Running SMART - CORE')
	parser = argparse.ArgumentParser()
	parser.add_argument('-strategyType', default='assessment', type=str)
	parser.add_argument('-encodingType', default='tfidf', type=str)
	parser.add_argument('-clusteringType', default='first', type=str)
	parser.add_argument('-clusters', default='150', type=str)
	parser.add_argument('-outputFolder',  default='output', type=str)
	parser.add_argument('-n_run', default='', type=str)
	parser.add_argument('-course', default='oli_intro_bio', type=str)
	parser.add_argument('-merge', default='off', type=str)
	parser.add_argument('-noClusterKeywordMapping',  action='store_false')
	parser.add_argument('-noAssessmentSkillMapping', action='store_false')
	parser.add_argument('-noParagraphSkillMapping',  action='store_false')

	args = parser.parse_args()

	strategyType = args.strategyType.lower()  # string
	encodingType = args.encodingType.lower()  # string
	clusteringType = args.clusteringType.lower()  # string
	clusters = args.clusters  # string
	if clusters == 'None':
		clusters = None
	course = args.course.lower()
	merge_status = args.merge.lower()
	outputFolder = args.outputFolder  # string
	n_run = args.n_run

	# switch arguments
	save_clusterKeywordMapping = args.noClusterKeywordMapping  # switch
	save_assessmentSkillMapping = args.noAssessmentSkillMapping  # switch
	save_paragraphSkillMapping = args.noParagraphSkillMapping  # switch

	# inputFolder = os.path.join(str(Path.home()),'Documents/OneDrive/PASTEL Project/SMART/assessment_paragraph data/intro_bio (with periods)') #macbook
	# inputFolder = os.path.join(str(Path.home()),'OneDrive/PASTEL Project/SMART/assessment_paragraph data/intro_bio (with periods)') #nerve, kona, mocha

	if course == 'oli_intro_bio':
		inputFolder = 'SMART_CORE/OneDrive-2020-12-04/intro_bio (with periods)_labelled'  # local (temporary) - OLI Biology
	elif course == 'oli_gen_chem':
		inputFolder = 'SMART_CORE/OneDrive-2020-12-04/gen_chem'  # local (temporary) - OLI General Chemistry I
	else:
		print("Invalid input for course title.")
		sys.exit()
	curr_dir = os.path.dirname(os.path.realpath(__file__))
	inputFolder = os.path.join(curr_dir, inputFolder)
	# print(inputFolder)
	# sys.exit()

	# print "strategy", strategy, "clustering", clustering, "encoding", encoding, "num_clusters", int(num_clusters), "save_clusterKeywordMapping", save_clusterKeywordMapping, "save_assessmentSkillMapping", save_assessmentSkillMapping, "save_paragraphSkillMapping", save_paragraphSkillMapping
	parameters = (strategyType, encodingType, clusteringType, clusters, save_clusterKeywordMapping, save_assessmentSkillMapping, save_paragraphSkillMapping, inputFolder, outputFolder, n_run, course, merge_status)
	text2_skill_mapping(parameters)


if __name__ == '__main__':
	main()
