import os
import re
import csv
from collections import defaultdict

"""
Using local (modified) summa package
"""
from .summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from .summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word

def get_wordCounts(input1_csv, input2_csv):
    wordCounts = defaultdict(int)

    with open(input1_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
        #     text = re.sub(r'[^\w\s]', '', row[2]) 
        #     for word in text.split():
        #         wordCounts[word] += 1
            text = row[2] #instructional text
            tokens = _clean_text_by_word(text, "english") #tokens (a Syntactic object) is a mapping (dict) of original_word to [original_word, lemma]
            for word in list(_tokenize_by_word(text)):
                if word in tokens:
                # if True:
                    wordCounts[word] += 1

    with open(input1_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
        #     text = re.sub(r'[^\w\s]', '', row[2]) 
        #     for word in text.split():
        #         wordCounts[word] += 1
    
            text = row[2] #instructional text
            tokens = _clean_text_by_word(text, "english") #tokens (a Syntactic object) is a mapping (dict) of original_word to [original_word, lemma]
            for word in list(_tokenize_by_word(text)):
                if word in tokens:
                # if True:
                    wordCounts[word] += 1

    return wordCounts
     

if __name__ == '__main__':
    inputFolder = 'OneDrive-2020-12-04/intro_bio (with periods)_labelled' #temporary
    input1_csv = os.path.join(inputFolder, 'assessments.csv')
    input2_csv = os.path.join(inputFolder, 'paragraphs.csv')
    word_counts = get_wordCounts(input1_csv, input2_csv)
    
    with open('word_counts.csv', 'w') as write_file:
        writer = csv.writer(write_file)
        for k, v in word_counts.items():
            writer.writerow([k,v])