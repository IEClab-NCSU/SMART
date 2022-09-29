from nltk.tokenize import sent_tokenize 
from sentence_transformers import SentenceTransformer
import numpy as np

#input: list of documents
# output: numpy matrix of embeddings of all the documents

def get_average_embedding(documents):
	model = SentenceTransformer('distilbert-base-nli-mean-tokens')
	doc_embeds = []
	#for each document, we will average the embedding of individual sentences
	for document in documents:
		# split document into individual sentences, get their embedding and average them
		sentences = sent_tokenize(document)

		vectors = model.encode(sentences) #For bert, we feed the input as a list of texts	
		doc_embed = np.mean(vectors, axis=0) #averaging all the embeddings
		doc_embeds.append(doc_embed)

	return np.array(doc_embeds)

#testing
if __name__ == '__main__':
	dummy_text = ['Although topic models such as LDA and NMF have shown to be good starting points, I always felt it took quite some effort through hyperparameter tuning to create meaningful topics. Moreover, I wanted to use transformer-based models such as BERT as they have shown amazing results in various NLP tasks over the last few years. Pre-trained models are especially helpful as they are supposed to contain more accurate representations of words and sentences.',
				'The great advantage of Doc2Vec is that the resulting document- and word embeddings are jointly embedding in the same space which allows document embeddings to be represented by nearby word embeddings. Unfortunately, this proved to be difficult as BERT embeddings are token-based and do not necessarily occupy the same space']
	embeddings = get_average_embedding(dummy_text)
	print(embeddings.shape)
	# print(embeddings)