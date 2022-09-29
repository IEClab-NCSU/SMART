""" Generate fine-tuned embeddings from the the fine-tuned BERT model
"""

from transformers import BertModel, BertConfig, BertTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

#input: list of documents
# output: numpy matrix of embeddings of all the documents
# Approach: Use a model that was trained on a classification task using the top-level sections from the original KC
# model (see /SMART_CORE/bert_finetuning_classification.py)
def get_fineTuned_embeddings(texts):
	print('Using Fine-tuned BERT')
	model_path='SMART_CORE/bert fine-tuning/bert-base-uncased_intro_bio'

	config = BertConfig.from_pretrained(model_path, output_hidden_states=True)
	model = BertModel.from_pretrained(model_path, config=config)
	tokenizer = BertTokenizer.from_pretrained(model_path)

	text_embeddings = []

	for text in texts:
		inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")

		outputs = model(**inputs) # outputs: [last_hidden_state, pooler_output, hidden_states]

		# print('len(outputs)', len(outputs))  # 3
		# print(outputs[0].shape)
		embedding = outputs[0][:,0,:]
		# print('temp.size()', temp.size())

		# hidden_states = outputs[2]
		# print('len(hidden_states)', len(hidden_states))  # 13

		# embedding_output = hidden_states[0]
		# print('embedding_output.size()', embedding_output.size())

		# print('outputs.pooler_output.size()', outputs.pooler_output.size())
		emb = embedding.detach().numpy()
		text_embeddings.append(np.squeeze(emb))
		# print(np.squeeze(emb).shape)
	
	return np.array(text_embeddings)

# Approach: Use a model that was trained on a classification task using the top-level sections from the original KC
# model (see /SMART_CORE/bert_finetuning_classification.py) and use chunking of the text and the mean embedding instead
# of truncation to obtain the paragraph / assessment / cluster embedding.
def get_fineTuned_embeddings_updated(texts):
	print('Using Updated Fine-tuned BERT')
	model_path = 'SMART_CORE/bert-base-uncased_intro_bio'

	config = BertConfig.from_pretrained(model_path, output_hidden_states=True)
	model = BertModel.from_pretrained(model_path, config=config)
	tokenizer = BertTokenizer.from_pretrained(model_path)

	text_embeddings = []

	for text in texts:
		tokens = tokenizer(text)

		sum_embeddings = np.array(np.zeros(768))
		count = 0
		max_tokens = 510
		length_tokens = len(tokens['input_ids']) - 1

		for j in range(1, length_tokens, max_tokens):
			if j + 254 < length_tokens:
				new_tokens = {'input_ids': tokens['input_ids'][j:j + max_tokens],
							  'token_type_ids': tokens['token_type_ids'][j:j + max_tokens],
							  'attention_mask': tokens['attention_mask'][j:j + max_tokens]}
			else:
				new_tokens = {'input_ids': tokens['input_ids'][j:length_tokens],
							  'token_type_ids': tokens['token_type_ids'][j:length_tokens],
							  'attention_mask': tokens['attention_mask'][j:length_tokens]}
			# get outputs from new_tokens
			new_tokens['input_ids'] = torch.as_tensor([new_tokens['input_ids']])
			new_tokens['token_type_ids'] = torch.as_tensor([new_tokens['token_type_ids']])
			new_tokens['attention_mask'] = torch.as_tensor([new_tokens['attention_mask']])

			outputs = model(**new_tokens)
			embedding = outputs[0][:, 0, :]
			emb = embedding.detach().numpy()
			emb = np.squeeze(emb)

			sum_embeddings += emb
			count += 1

		# average the embeddings
		average_emb = sum_embeddings / count

		text_embeddings.append(average_emb)

	return np.array(text_embeddings)


# Approach: Use a pre-trained Sentence-BERT model to obtain the paragraph/assessment/cluster embeddings.
def get_embeddings_sentBert(texts):
	print('Using Sentence-BERT')
	model = SentenceTransformer('paraphrase-distilroberta-base-v2')

	text_embeddings = model.encode(texts)

	return np.array(text_embeddings)

# Approach: Use a pre-trained BERT model to obtain the paragraph/assessment/cluster embeddings.
# Note: This is the approach currently in use (text2_skill_mapping.py and clustering.py) since it demonstrated the
# best performance of the options for the following hyperparameters: Strategy - assessment, Cluster Method - first,
# Number of Clusters - 100.
def get_embeddings_pretrainBERT(texts):
	print('Using Pre-Trained BERT')
	model_path = 'bert-base-uncased'
	model = BertModel.from_pretrained(model_path, output_hidden_states=True)
	tokenizer = BertTokenizer.from_pretrained(model_path)

	text_embeddings = []

	for text in texts:
		tokens = tokenizer(text)

		sum_embeddings = np.array(np.zeros(768))
		count = 0
		max_tokens = 510
		length_tokens = len(tokens['input_ids']) - 1

		for j in range(1, length_tokens, max_tokens):
			if j + 254 < length_tokens:
				new_tokens = {'input_ids': tokens['input_ids'][j:j + max_tokens],
							  'token_type_ids': tokens['token_type_ids'][j:j + max_tokens],
							  'attention_mask': tokens['attention_mask'][j:j + max_tokens]}
			else:
				new_tokens = {'input_ids': tokens['input_ids'][j:length_tokens],
							  'token_type_ids': tokens['token_type_ids'][j:length_tokens],
							  'attention_mask': tokens['attention_mask'][j:length_tokens]}
			# get outputs from new_tokens
			new_tokens['input_ids'] = torch.as_tensor([new_tokens['input_ids']])
			new_tokens['token_type_ids'] = torch.as_tensor([new_tokens['token_type_ids']])
			new_tokens['attention_mask'] = torch.as_tensor([new_tokens['attention_mask']])

			outputs = model(**new_tokens)
			embedding = outputs[0][:, 0, :]
			emb = embedding.detach().numpy()
			emb = np.squeeze(emb)

			sum_embeddings += emb
			count += 1

		# average the embeddings
		average_emb = sum_embeddings / count

		text_embeddings.append(average_emb)

	return np.array(text_embeddings)

# Approach: Use a pre-trained BERT model with additional pre-training using the paragraphs/assessments
# to obtain the paragraph/assessment/cluster embeddings (see bert_continue_pretraining.py).
def get_embeddings_custom_pretrainBERT(texts):
	print('Using Custom Pre-Trained BERT')
	model_path = 'SMART_CORE/./bert-continue-pretrain/continue_pretrain/'
	model = BertModel.from_pretrained(model_path, output_hidden_states=True)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	text_embeddings = []

	for text in texts:
		tokens = tokenizer(text)

		sum_embeddings = np.array(np.zeros(768))
		count = 0
		max_tokens = 510
		length_tokens = len(tokens['input_ids']) - 1

		for j in range(1, length_tokens, max_tokens):
			if j + 254 < length_tokens:
				new_tokens = {'input_ids': tokens['input_ids'][j:j + max_tokens],
							  'token_type_ids': tokens['token_type_ids'][j:j + max_tokens],
							  'attention_mask': tokens['attention_mask'][j:j + max_tokens]}
			else:
				new_tokens = {'input_ids': tokens['input_ids'][j:length_tokens],
							  'token_type_ids': tokens['token_type_ids'][j:length_tokens],
							  'attention_mask': tokens['attention_mask'][j:length_tokens]}
			# get outputs from new_tokens
			new_tokens['input_ids'] = torch.as_tensor([new_tokens['input_ids']])
			new_tokens['token_type_ids'] = torch.as_tensor([new_tokens['token_type_ids']])
			new_tokens['attention_mask'] = torch.as_tensor([new_tokens['attention_mask']])

			outputs = model(**new_tokens)
			embedding = outputs[0][:, 0, :]
			emb = embedding.detach().numpy()
			emb = np.squeeze(emb)

			sum_embeddings += emb
			count += 1

		# average the embeddings
		average_emb = sum_embeddings / count

		text_embeddings.append(average_emb)

	return np.array(text_embeddings)


if __name__ == '__main__':
	dummy_text = ['Although topic models such as LDA and NMF have shown to be good starting points.','This proved to be difficult as BERT embeddings are token-based and do not necessarily occupy the same space']	
	# print('len(dummy_text)', len(dummy_text))

	embeddings = get_fineTuned_embeddings(dummy_text)
	print('embeddings.shape', embeddings.shape)
	# print('embeddings', embeddings)