from datasets import load_dataset
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import pandas as pd
import numpy as np
import nltk
import string
import shutil
import sys
import os
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def prep_folders():
    dataframe_path = 'Dataframes/'
    training_data_path = 'TrainingDatasets/'
    validation_data_path = 'ValidationDatasets/'

    if os.path.exists(dataframe_path):
        shutil.rmtree(dataframe_path)
    if os.path.exists(training_data_path):
        shutil.rmtree(training_data_path)
    if os.path.exists(validation_data_path):
        shutil.rmtree(validation_data_path)

    os.mkdir(dataframe_path)
    os.mkdir(training_data_path)
    os.mkdir(validation_data_path)


def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    new_text = " ".join([lemmatizer.lemmatize(word, get_pos(word)) for word in word_tokenize(text)])
    return new_text


def remove_stopwords(text):
    stop_words = stopwords.words('english') + ['correct', 'true', 'false', 'yes', 'following', 'mathrm']
    new_text = [word for word in word_tokenize(text) if word not in stop_words]
    new_text = " ".join(new_text)
    return new_text


def stem(text):
    ps = PorterStemmer()
    new_text = " ".join([ps.stem(word) for word in word_tokenize(text)])
    return new_text


def preprocess(text):
    # apply lowercasing
    new_text = text.lower()
    # remove punctuation and numeric characters

    # New
    new_text = re.sub(r"[^\s]+_[^\s]+", "", new_text)
    # new_text = re.sub(r"[^\s]+-[^\s]+", "", new_text)
    new_text = re.sub(r"[(][^)]*[)]", "", new_text)
    new_text = new_text.replace('_', '.')
    new_text = new_text.replace('-', ' ')
    new_text = re.sub(r"[^a-zA-Z\s]", "", new_text)

    # remove extra whitespace
    new_text = " ".join(new_text.split())
    return new_text


def prepare_data_hf(dataset_path, df_format='full'):
    dataset = load_dataset(dataset_path, 'raw')
    dataset.set_format(type='pandas')

    num_documents = len(dataset['test']['document'])
    data_df = dataset['test'][0:num_documents]

    # join the list of words for each document to make a complete string for each document.
    data_df['whole_document'] = data_df['document'].apply(lambda x: " ".join(x))

    # apply lower casing, strip numeric text, strip punctuation, removal of excess whitespace, and lemmatization
    # for each document.
    data_df['processed_whole_document'] = data_df['whole_document'].apply(lambda text: preprocess(text))
    # data_df['processed_whole_document'] = data_df['processed_whole_document'].apply(lambda text: lemmatize(text))
    data_df['processed_whole_document'] = data_df['processed_whole_document'].apply(lambda text: remove_stopwords(text))

    # join the lists of keyphrases for each document to make one list of gold keyphrases.
    data_df['keyphrases'] = data_df['extractive_keyphrases'].apply(lambda x: x.tolist()) + data_df['abstractive_keyphrases'].apply(lambda x: x.tolist())
    data_df['keyphrases'] = data_df['keyphrases'].apply(lambda kp_list: np.array(kp_list))
    data_df['eval_processed_keyphrases'] = data_df['keyphrases'].apply(lambda kp_list: [preprocess(keyphrase) for keyphrase in kp_list])
    # data_df['eval_processed_keyphrases'] = data_df['processed_keyphrases'].apply(lambda kp_list: [lemmatize(keyphrase) for keyphrase in kp_list])
    # data_df['eval_processed_keyphrases'] = data_df['eval_processed_keyphrases'].apply(lambda kp_list: [remove_stopwords(keyphrase) for keyphrase in kp_list])

    if df_format == 'first-only':
        # keep only the first keyphrase listed
        data_df['keyphrases'] = data_df['keyphrases'].apply(lambda kp_list: [kp_list[0]])
        data_df['eval_processed_keyphrases'] = data_df['eval_processed_keyphrases'].apply(lambda kp_list: [kp_list[0]])

    # format the document text and keyphrases into a combined string for finetuning GPT-2
    data_df['condensed_processed_whole_document'] = data_df['processed_whole_document'].apply(
        lambda text: " ".join(word_tokenize(text)[:325]))
    special_token = '\n<|endoftext|>'
    data_df['combined'] = 'Document: ' + data_df.condensed_processed_whole_document + '\nKeyphrases:\n' + data_df.eval_processed_keyphrases.str.join(
        ',') + special_token

    return data_df


def prepare_data_oli(dataset_path, oli_labels='verbose', verbose=False):

    if 'oli_intro_bio' in dataset_path:
        assessments_path = os.path.join(str(Path.home()),
                                        'OneDrive/SMART/assessment_paragraph data/intro_bio (with periods)_labelled/assessments.csv')
        datashop_skill_model_path = '../intro_bio_skill_map.tsv'
        datashop_skill_description = '../intro_bio_skills.tsv'
        drop_columns = []
    elif 'oli_gen_chem' in dataset_path:
        assessments_path = os.path.join(str(Path.home()),
                                        'OneDrive/SMART/assessment_paragraph data/gen_chem/assessments.csv')
        datashop_skill_model_path = '../chem1_skill_map_2.3_5_20_20-Problems.tsv'
        datashop_skill_description = '../chem1_skill_map_2.3_5_20_20-Skills.tsv'
        drop_columns = ['Step', 'Skill2', 'Skill3', 'Skill4', 'Skill5', 'Skill6']
    else:
        print("Invalid input for course.")
        sys.exit()

    # read the problem title and full text of the assessment item from file
    full_problems = pd.read_csv(assessments_path, names=['Section', 'Problem', 'Full_Problem_Text'])
    full_problems = full_problems.drop('Section', axis=1)

    # read the problem to skill mapping from Datashop file
    problem_skill_map = pd.read_csv(datashop_skill_model_path, skiprows=1, sep='\t')
    problem_skill_map = problem_skill_map.drop(drop_columns, axis=1)
    problem_skill_map['Problem'] = problem_skill_map['Resource'] + '_' + problem_skill_map['Problem']
    problem_skill_map = problem_skill_map.drop('Resource', axis=1)
    problem_skill_map.rename(columns={'Skill1': 'Human_Skill'}, inplace=True)

    # read the skill title and full skill description from Datashop file
    full_human_skills = pd.read_csv(datashop_skill_description, sep='\t')
    full_human_skills = full_human_skills.drop(['p', 'gamma0', 'gamma1', 'lambda0'], axis=1)
    full_human_skills.rename(columns={'Skill': 'Human_Skill', 'Title': 'Full_Human_Skill_Description'},
                             inplace=True)

    # combine the full problems with the problem to skill map
    abbr_df = full_problems.merge(problem_skill_map[['Problem', 'Human_Skill']])

    # combine the abbr_df with full human skills to add the verbose skill descriptions
    full_df = abbr_df.merge(full_human_skills[['Human_Skill', 'Full_Human_Skill_Description']])
    full_df['Human_Skill'] = full_df['Human_Skill'].str.replace('_', ' ')
    full_df['Full_Human_Skill_Description'] = full_df['Full_Human_Skill_Description'].apply(
        lambda text: preprocess(text))

    if oli_labels == 'short':
        unique_human_skills = full_df['Human_Skill'].unique()
        # print(len(unique_human_skills))
    elif oli_labels == 'verbose':
        unique_human_skills = full_df['Full_Human_Skill_Description'].unique()
        # print(unique_human_skills)
        # print(len(unique_human_skills))
    else:
        print(
            "Invalid use of flag for using shortened or verbose form of the human-generated skill labels for the OLI courseware.")
        sys.exit()

    clusterIndex_to_clusteredText = {}

    for cluster_index in range(0, len(unique_human_skills)):
        if oli_labels == 'short':
            cluster_df = full_df.loc[full_df['Human_Skill'] == unique_human_skills[cluster_index]]
            full_df.loc[full_df['Human_Skill'] == unique_human_skills[cluster_index], 'Cluster_Index'] = cluster_index
        else:
            cluster_df = full_df.loc[full_df['Full_Human_Skill_Description'] == unique_human_skills[cluster_index]]
            full_df.loc[
                full_df['Full_Human_Skill_Description'] == unique_human_skills[
                    cluster_index], 'Cluster_Index'] = cluster_index
        clusterText_list = cluster_df['Full_Problem_Text'].values.tolist()
        clusterText = ''
        for item_index in range(0, len(clusterText_list)):
            clusterText += clusterText_list[item_index] + '. '
        clusterIndex_to_clusteredText[cluster_index] = clusterText

    # full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    full_df['Cluster_Index'] = full_df['Cluster_Index'].astype(int)

    orig_documents_whole = list(clusterIndex_to_clusteredText.values())

    # apply lower casing, strip numeric text, strip punctuation, and remove excess whitespace from the text
    # for each document.
    pattern = r'{.*}'
    interim_processed_documents_whole = [re.sub(pattern, "", text) for text in orig_documents_whole]
    pattern = r'\$\$.*\$\$'
    interim_processed_documents_whole = [re.sub(pattern, "", text) for text in interim_processed_documents_whole]
    pattern = r'\\\\[\w\W]*}'
    whole_documents = [re.sub(pattern, "", text) for text in interim_processed_documents_whole]
    processed_whole_documents = [preprocess(text) for text in whole_documents]
    # processed_whole_documents = [lemmatize(text) for text in processed_whole_documents]
    processed_whole_documents = [remove_stopwords(text) for text in processed_whole_documents]

    gold_keyphrase_list = []
    eval_processed_gold_keyphrase_list = []
    for text in unique_human_skills:
        # print(f'Skill: {text}')
        processed_text = preprocess(text)
        # eval_processed_text = lemmatize(processed_text)
        # eval_processed_text = remove_stopwords(eval_processed_text)
        # print(f'Processed Skill: {processed_text}')
        gold_keyphrase_list.append([text])
        eval_processed_gold_keyphrase_list.append([processed_text])

    oli_df = pd.DataFrame()
    oli_df['whole_document'] = whole_documents
    oli_df['processed_whole_document'] = processed_whole_documents
    oli_df['condensed_processed_whole_document'] = oli_df['processed_whole_document'].apply(lambda text: " ".join(word_tokenize(text)[:300]))
    oli_df['keyphrases'] = gold_keyphrase_list
    oli_df['eval_processed_keyphrases'] = eval_processed_gold_keyphrase_list

    # format the document text and keyphrases into a combined string for processing by GPT-2 model
    special_token = '\n<|endoftext|>'
    oli_df['combined'] = 'Document:\n' + oli_df.condensed_processed_whole_document + '\nKeyphrases:\n' + oli_df.eval_processed_keyphrases.str.join(
        ',') + special_token

    if verbose:
        print('Dataset Retrieval is complete.')

    return oli_df


def main():
    prep_folders()
    # pd.set_option('display.max_columns', 500)

    kdd_dataset_path = 'midas/kdd'
    inspec_dataset_path = 'midas/inspec'
    oli_intro_bio_dataset_path = 'oli_intro_bio'
    oli_gen_chem_dataset_path = 'oli_gen_chem'

    kdd_first_only_data_df = prepare_data_hf(kdd_dataset_path, df_format='first-only')
    inspec_first_only_data_df = prepare_data_hf(inspec_dataset_path, df_format='first-only')
    inspec_kdd_first_only_df = pd.concat([inspec_first_only_data_df, kdd_first_only_data_df])
    oli_intro_bio_df = prepare_data_oli(oli_intro_bio_dataset_path)
    oli_gen_chem_df = prepare_data_oli(oli_gen_chem_dataset_path)

    # create the train, validation, and test datasets
    train_data_bio_ten_percent_df = oli_intro_bio_df.sample(frac=0.1, random_state=25)
    other_data_bio_ten_percent_df = oli_intro_bio_df.drop(train_data_bio_ten_percent_df.index)
    val_data_bio_ten_percent_df = other_data_bio_ten_percent_df.sample(frac=0.105, random_state=18)
    test_data_bio_eighty_percent_df = other_data_bio_ten_percent_df.drop(val_data_bio_ten_percent_df.index)

    train_data_chem_ten_percent_df = oli_gen_chem_df.sample(frac=0.1, random_state=25)
    other_data_chem_ten_percent_df = oli_gen_chem_df.drop(train_data_chem_ten_percent_df.index)
    val_data_chem_ten_percent_df = other_data_chem_ten_percent_df.sample(frac=0.105, random_state=18)
    test_data_chem_eighty_percent_df = other_data_chem_ten_percent_df.drop(val_data_chem_ten_percent_df.index)

    train_inspec_kdd_first_only_df = inspec_kdd_first_only_df.sample(frac=0.8, random_state=25)
    other_inspec_kdd_first_only_df = inspec_kdd_first_only_df.drop(train_inspec_kdd_first_only_df.index)
    val_inspec_kdd_first_only_df = other_inspec_kdd_first_only_df.sample(frac=0.5, random_state=18)
    test_inspec_kdd_first_only_df = other_inspec_kdd_first_only_df.drop(val_inspec_kdd_first_only_df.index)

    # save the train, validation, and test datasets
    train_bio_10_percent = train_data_bio_ten_percent_df.combined.values
    with open('TrainingDatasets/train_bio_10_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(train_bio_10_percent))

    val_bio_10_percent = val_data_bio_ten_percent_df.combined.values
    with open('ValidationDatasets/val_bio_10_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(val_bio_10_percent))

    test_bio_80_percent = test_data_bio_eighty_percent_df.combined.values
    with open('ValidationDatasets/test_bio_80_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(test_bio_80_percent))

    test_bio_full = oli_intro_bio_df.combined.values
    with open('ValidationDatasets/test_bio_full.txt', 'w') as filepath:
        filepath.write('\n'.join(test_bio_full))

    test_chem_full = oli_gen_chem_df.combined.values
    with open('ValidationDatasets/test_chem_full.txt', 'w') as filepath:
        filepath.write('\n'.join(test_chem_full))

    train_chem_10_percent = train_data_chem_ten_percent_df.combined.values
    with open('TrainingDatasets/train_chem_10_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(train_chem_10_percent))

    val_chem_10_percent = val_data_chem_ten_percent_df.combined.values
    with open('ValidationDatasets/val_chem_10_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(val_chem_10_percent))

    test_chem_80_percent = test_data_chem_eighty_percent_df.combined.values
    with open('ValidationDatasets/test_chem_80_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(test_chem_80_percent))

    train_inspec_kdd_first_only_80_percent = train_inspec_kdd_first_only_df.combined.values
    with open('TrainingDatasets/train_inspec_kdd_first_only_80_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(train_inspec_kdd_first_only_80_percent))

    val_inspec_kdd_first_only_10_percent = val_inspec_kdd_first_only_df.combined.values
    with open('ValidationDatasets/val_inspec_kdd_first_only_10_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(val_inspec_kdd_first_only_10_percent))

    test_inspec_kdd_first_only_10_percent = test_inspec_kdd_first_only_df.combined.values
    with open('ValidationDatasets/test_inspec_kdd_first_only_10_percent.txt', 'w') as filepath:
        filepath.write('\n'.join(test_inspec_kdd_first_only_10_percent))

    # save the full dataframes for each dataset
    kdd_first_only_data_df.to_pickle('Dataframes/kdd_first_only_dataset_preprocessed.pkl')
    inspec_first_only_data_df.to_pickle('Dataframes/inspec_first_only_dataset_preprocessed.pkl')
    inspec_kdd_first_only_df.to_pickle('Dataframes/inspec_kdd_first_only_dataset_preprocessed.pkl')
    oli_intro_bio_df.to_pickle('Dataframes/oli_intro_bio_dataset_preprocessed.pkl')
    oli_gen_chem_df.to_pickle('Dataframes/oli_gen_chem_dataset_preprocessed.pkl')

    # save the train, validation, and test dataframes for OLI
    train_data_bio_ten_percent_df.to_pickle('Dataframes/train_bio_10_percent.pkl')
    train_data_chem_ten_percent_df.to_pickle('Dataframes/train_chem_10_percent.pkl')

    val_data_bio_ten_percent_df.to_pickle('Dataframes/val_bio_10_percent.pkl')
    val_data_chem_ten_percent_df.to_pickle('Dataframes/val_chem_10_percent.pkl')

    test_data_bio_eighty_percent_df.to_pickle('Dataframes/test_bio_80_percent.pkl')
    test_data_chem_eighty_percent_df.to_pickle('Dataframes/test_chem_80_percent.pkl')

    train_inspec_kdd_first_only_df.to_pickle('Dataframes/train_inspec_kdd_first_only.pkl')
    val_inspec_kdd_first_only_df.to_pickle('Dataframes/val_inspec_kdd_first_only.pkl')
    test_inspec_kdd_first_only_df.to_pickle('Dataframes/test_inspec_kdd_first_only.pkl')


if __name__ == '__main__':
    main()
