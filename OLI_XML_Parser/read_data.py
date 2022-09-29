from pathlib import Path
import os
import csv
import random


def read_data():
	# inputFolder = '../OneDrive-2020-12-04/intro_bio (with periods)_labelled' #temporary
	inputFolder = os.path.join(str(Path.home()),'Documents/OneDrive/PASTEL Project/SMART/assessment_paragraph data/intro_bio (with periods)_labelled/')

	labels, ids, texts = [], [], []
	target_names = ['_u800_Bio_Intro','_u810_BioChemistry','cell_repro_and_genetics','gene_regulation','phys_prop_cell','redox','stored_info_dna_protein','struc_fun','transformations_energy_matter']
	target_to_id = {name:id for id, name in enumerate(target_names)}
	with open(os.path.join(inputFolder, 'assessments.csv'), 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			labels.append(target_to_id[row[0]])
			# ids.append(row[1])
			texts.append(row[2])
  
	with open(os.path.join(inputFolder, 'paragraphs.csv'), 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			labels.append(target_to_id[row[0]])
			# ids.append(row[1])
			texts.append(row[2])

	labels_texts = list(zip(labels, texts))
	random.shuffle(labels_texts)
	labels, texts = zip(*labels_texts)

	return texts, labels, target_names

def getLengths(texts):
	max_words = 0
	total_words = 0
	word_counts = []
	for text in texts:
		n = len(text.split())
		max_words = max(max_words, n)
		total_words += n
		word_counts.append(n)
	
	print('max_words ', max_words, 'avg_words ', total_words/len(texts))
	print(sorted(word_counts, reverse = True))
	import matplotlib.pyplot as plt
	plt.hist(word_counts)
	plt.show()

if __name__ == '__main__':
	texts, labels, target_names  = read_data()
	print('len(texts)', len(texts))
	print('len(texts)', len(labels))
	print('len(target_names) ', len(target_names))

	getLengths(texts)