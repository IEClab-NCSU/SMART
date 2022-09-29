'''
Clean the texts using tokenization and lemmatization
'''

import os
import re

from SMART_CORE.summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from SMART_CORE.summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word
from SMART_CORE.custom_stopwords import get_custom_stopwords

def lemmatize(result):
    try:
        text = ' '.join(result.values())
    except TypeError:
        text = ''

    custom_stopwords = get_custom_stopwords() #these are additional stopwords that are abundant in the course text but are uninformative for SMART
    lemmatized_text = []
    tokens = _clean_text_by_word(text, "english", additional_stopwords=custom_stopwords) #tokens (a Syntactic object) is a mapping (dict) of original_word to [original_word, lemma]
    for word in list(_tokenize_by_word(text)):
        if word in tokens:
            lemmatized_text.append(tokens[word].token)

    lemmatized_text.append('.')
    return (' ').join(lemmatized_text)

def clean_text(texts):
	return [lemmatize(text) for text in texts]