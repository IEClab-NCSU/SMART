"""Perform preprocessing on the input texts and return them
"""
import os
import re
import csv

"""
Using local (modified) summa package
"""
from .summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from .summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word
from .custom_stopwords import get_custom_stopwords


def lemmatize(text):
	lemmatized_text = []
	tokens = _clean_text_by_word(text, "english", additional_stopwords=get_custom_stopwords()) #tokens (a Syntactic object) is a mapping (dict) of original_word to [original_word, lemma]
	for word in list(_tokenize_by_word(text)):
		if word in tokens:
			lemmatized_text.append(tokens[word].token)

	lemmatized_text.append('.')
	lemmatized_text = (' ').join(lemmatized_text)	
	return lemmatized_text

def getText_Textname(filename):
	text_ids = []
	lemmatized_texts = []
	original_texts = []

	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			text_ids.append(row[1]) #instructional textid

			text = row[2] #instructional text
			
			original_texts.append(text)
			lemmatized_texts.append(lemmatize(text))

	return text_ids, lemmatized_texts, original_texts

def create_input_lists_from_csv(input1_csv, input2_csv):
	# Inputs:-
	# input1 = csv filename corresponding to mapbase
	# input2 = folder filename corresponding to mapping texts
	# Outputs:-
	# text1, text1name, text2, text2name = respective texts and textfile names

	text_ids1, lemmatized_texts1, original_texts1 = getText_Textname(input1_csv)
	text_ids2, lemmatized_texts2, original_texts2 = getText_Textname(input2_csv)
	return text_ids1, lemmatized_texts1, original_texts1, text_ids2, lemmatized_texts2, original_texts2
