from sentence_transformers import SentenceTransformer
import csv 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import nltk
nltk.download('punkt')
from nltk import sent_tokenize

def	interParagraphSimilarity(texts, strategyType):
	texts = texts[:6]
	n = len(texts)
	
	all_sentences = []
	ranges = []
	counter = 0
	for text in texts:
		sentences = sent_tokenize(text)
		beg = counter
		if sentences:
			for sentence in sentences:
				all_sentences.append(sentence)
				counter += 1
		else:
			all_sentences.append('')
			counter += 1
		ranges.append((beg, counter))

	embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
	vectors = embedder.encode(all_sentences) 	
	
	interParagraph_similarityMatrix = [[None]*n for _ in range(n)]
	for i in range(n):
		first, last = ranges[i]
		para1 = vectors[first:last]
		print('####')
		print(i)
		print(all_sentences[first:last])
		for j in range(i, n):
			first, last = ranges[j]
			para2 = vectors[first:last]

			print(j)
			print(all_sentences[first:last])

			interParagraph_similarityMatrix[i][j] = np.mean(cosine_similarity(para1, para2))
			
		print('Done',i)

	file = 'interParagraph_similarityMatrix_sample_{}.csv'.format(strategyType)
	with open(file, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(interParagraph_similarityMatrix) 

	sys.exit()
