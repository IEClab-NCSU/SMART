import os
from create_input_lists import create_input_lists_from_csv

input1 = 'assessments.csv'
input2 = 'paragraphs.csv'

inputFolder = './OneDrive-2020-12-04/intro_bio (with periods)_labelled'  # local (temporary)

input1_path = os.path.join(inputFolder, input1)
input2_path = os.path.join(inputFolder, input2)

text_ids1, lemmatized_texts1, original_texts1, text_ids2, lemmatized_texts2, original_texts2 = create_input_lists_from_csv(input1_path, input2_path)

# output each paragraph and assessment as a single line in a text file
with open('./bert-continue-pretrain/raw-text-for-bert.txt', 'w') as output:
    for text in original_texts1:
        output.write(text+'\n')
    for text in original_texts2:
        output.write(text+'\n')
