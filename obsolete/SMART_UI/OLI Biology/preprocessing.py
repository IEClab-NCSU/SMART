"""
Preprocess the text to replace new line, tab character and empty characters

"""

import re
from nltk import sent_tokenize


def preprocess(content, use_sentence):
    content = content.decode('utf-8')
    #print content
    #print use_sentence

    if use_sentence:
        list = sent_tokenize(content)
    else:
        list = content.split("\n\n")

    list = [re.sub(r'\n', ' ', l) for l in list]
    list = [re.sub(r'\t', ' ', l) for l in list]
    list = [re.sub(r' +', ' ', l) for l in list]
    list = filter(None, list)
    #list = [l.decode('utf-8') for l in list]
    return list
