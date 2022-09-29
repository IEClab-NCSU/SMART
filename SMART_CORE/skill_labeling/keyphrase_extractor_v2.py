from datasets import load_dataset
from pathlib import Path
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from rake_nltk import Rake
from yake import KeywordExtractor
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import sys
import os
import nltk
import re
import warnings
import torch
import pke
import random
random.seed(2022)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '2022'
np.random.seed(2022)
torch.manual_seed(2022)
warnings.filterwarnings('ignore')
# nltk.download('punkt')
# nltk.download('stopwords')
# !python -m nltk.downloader stopwords
# !python -m nltk.downloader universal_tagset
# !python -m spacy download en_core_web_sm # download the english model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from summa.keywords import keywords


class KeyphraseExtractor:
    def __init__(self, dataset_name, verbose=False, oli_labels='verbose', gpu_override=True):

        if not gpu_override:
            cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                print('Using Device: {}'.format(cuda_device))
            else:
                print('Using device:', self.device)
        else:
            self.device = torch.device('cpu')
            print('Using device:', self.device)

        self.dataset_name = dataset_name

        self.data_df = pd.DataFrame()

        self.original_documents_whole = []
        self.processed_documents_whole = []
        self.processed_documents_tokenized = []

        self.document_labels = []

        self.stop_words = stopwords.words('english') + ['correct', 'true', 'false', 'yes', 'following', 'mathrm']
        self.get_dataset(verbose, oli_labels)

        # summary statistics for the dataset
        self.num_docs = 0
        self.tokens_per_doc = 0.0
        self.gold_kp_per_doc = 0.0
        self.percent_stopwords_gold_kp = 0.0
        self.percent_kp_words_missing_from_doc = 0.0
        self.summarize_dataset(verbose)

        self.candidate_list = []
        self.cand_per_doc = 0.0
        self.percent_kp_missing_in_cand = 0.0

        self.document_embeddings = []
        self.unigram_candidate_embeddings = []
        self.other_candidate_embeddings = []

        self.score_dicts = []

        self.final_keyphrases = []
        self.final_keyphrase_scores = []

        self.overall_precision_partial = 0.0
        self.overall_recall_partial = 0.0
        self.overall_f1_partial = 0.0

        self.overall_precision_exact = 0.0
        self.overall_recall_exact = 0.0
        self.overall_f1_exact = 0.0

    def reinitialize(self):
        self.score_dicts = []

        self.final_keyphrases = []
        self.final_keyphrase_scores = []

        self.overall_precision_partial = 0.0
        self.overall_recall_partial = 0.0
        self.overall_f1_partial = 0.0

        self.overall_precision_exact = 0.0
        self.overall_recall_exact = 0.0
        self.overall_f1_exact = 0.0

    @staticmethod
    def get_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
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
        new_text = re.sub(r"[^a-zA-Z\s!.?]", "", new_text)

        # remove extra whitespace
        new_text = " ".join(new_text.split())
        return new_text

    @staticmethod
    def stem(text):
        ps = PorterStemmer()
        new_text = " ".join([ps.stem(word) for word in word_tokenize(text)])
        return new_text

    def lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()
        new_text = " ".join([lemmatizer.lemmatize(word, self.get_pos(word)) for word in word_tokenize(text)])
        return new_text

    def remove_stopwords(self, text):
        new_text = [word for word in word_tokenize(text) if word not in self.stop_words]
        new_text = " ".join(new_text)
        return new_text

    def valid_text(self, text):
        new_text = "".join([char for char in text if char not in ['.', '!', '?']])
        new_text = [word for word in word_tokenize(new_text) if word not in self.stop_words]
        new_text = " ".join(new_text)
        valid = False
        for word in word_tokenize(new_text):
            if len(word) >= 2:
                valid=True
        return  valid

    def get_dataset(self, verbose, oli_labels):

        if self.dataset_name in ['nus', 'krapivin', 'inspec', 'kdd', 'pubmed']:
            # retrieve dataset from Hugging Face
            dataset_path = 'midas/' + self.dataset_name
            dataset = load_dataset(dataset_path, 'raw')
            dataset.set_format(type='pandas')
            num_documents = len(dataset['test']['document'])

            # store data in pandas dataframe
            self.data_df = dataset['test'][0:num_documents]

            # join the list of words for each document to make a complete string for each document.
            # store result as a class variable.
            self.data_df['whole_document'] = self.data_df['document'].apply(lambda x: " ".join(x))
            self.original_documents_whole = self.data_df.whole_document.values

            # apply lower casing, strip numeric text, strip punctuation, removal of excess whitespace, and lemmatization
            # for each document.
            self.data_df['processed_document_whole'] = self.data_df['whole_document'].apply(lambda text: self.preprocess(text))
            self.processed_documents_whole = self.data_df.processed_document_whole.values

            # remove stopwords from the text for each document. store tokenized version and string version as class
            # variables.
            self.data_df['processed_document_tokenized'] = self.data_df['processed_document_whole'].apply(lambda text: [word for word in word_tokenize(self.remove_stopwords(text))])
            self.processed_documents_tokenized = self.data_df.processed_document_tokenized.values

            self.data_df['keyphrases'] = self.data_df['extractive_keyphrases'].apply(lambda x: x.tolist()) + self.data_df['abstractive_keyphrases'].apply(lambda x: x.tolist())
            self.data_df['keyphrases'] = self.data_df['keyphrases'].apply(lambda kp_list: np.array(kp_list))
            self.data_df['keyphrases'] = self.data_df['keyphrases'].apply(lambda kp_list: [self.preprocess(keyphrase) for keyphrase in kp_list])

            self.document_labels = self.data_df.keyphrases.values

        if self.dataset_name in ['oli-intro-bio', 'oli-gen-chem']:
            if self.dataset_name == 'oli-intro-bio':
                assessments_path = os.path.join(str(Path.home()), 'OneDrive/SMART/assessment_paragraph data/intro_bio (with periods)_labelled/assessments.csv')
                datashop_skill_model_path = 'intro_bio_skill_map.tsv'
                datashop_skill_description = 'intro_bio_skills.tsv'
                drop_columns = []
            elif self.dataset_name == 'oli-gen-chem':
                assessments_path = os.path.join(str(Path.home()),
                                                'OneDrive/SMART/assessment_paragraph data/gen_chem/assessments.csv')
                datashop_skill_model_path = 'chem1_skill_map_2.3_5_20_20-Problems.tsv'
                datashop_skill_description = 'chem1_skill_map_2.3_5_20_20-Skills.tsv'
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
            full_df['Full_Human_Skill_Description'] = full_df['Full_Human_Skill_Description'].apply(lambda text: self.preprocess(text))

            if oli_labels == 'short':
                unique_human_skills = full_df['Human_Skill'].unique()
            elif oli_labels == 'verbose':
                unique_human_skills = full_df['Full_Human_Skill_Description'].unique()
            else:
                print("Invalid use of flag for using shortened or verbose form of the human-generated skill labels for the OLI courseware.")
                sys.exit()

            clusterIndex_to_clusteredText = {}

            for cluster_index in range(0, len(unique_human_skills)):
                if oli_labels == 'short':
                    cluster_df = full_df.loc[full_df['Human_Skill'] == unique_human_skills[cluster_index]]
                    full_df.loc[full_df['Human_Skill'] == unique_human_skills[cluster_index], 'Cluster_Index'] = cluster_index
                else:
                    cluster_df = full_df.loc[full_df['Full_Human_Skill_Description'] == unique_human_skills[cluster_index]]
                    full_df.loc[
                        full_df['Full_Human_Skill_Description'] == unique_human_skills[cluster_index], 'Cluster_Index'] = cluster_index
                clusterText_list = cluster_df['Full_Problem_Text'].values.tolist()
                clusterText = ''
                for item_index in range(0, len(clusterText_list)):
                    clusterText += clusterText_list[item_index] + '. '
                clusterIndex_to_clusteredText[cluster_index] = clusterText

            full_df['Cluster_Index'] = full_df['Cluster_Index'].astype(int)

            orig_documents_whole = list(clusterIndex_to_clusteredText.values())

            # apply lower casing, strip numeric text, strip punctuation, and remove excess whitespace from the text
            # for each document.
            pattern = r'{.*}'
            interim_processed_documents_whole = [re.sub(pattern, "", text) for text in orig_documents_whole]
            pattern = r'\$\$.*\$\$'
            interim_processed_documents_whole = [re.sub(pattern, "", text) for text in interim_processed_documents_whole]
            pattern = r'\\\\[\w\W]*}'
            interim_processed_documents_whole = [re.sub(pattern, "", text) for text in interim_processed_documents_whole]
            self.original_documents_whole = interim_processed_documents_whole
            interim_processed_documents_whole = [text.lower() for text in interim_processed_documents_whole]
            interim_processed_documents_whole = [self.preprocess(text) for text in interim_processed_documents_whole]

            self.processed_documents_whole = interim_processed_documents_whole

            # remove stopwords from the text for each document. store tokenized version and string version as class
            # variables.
            self.processed_documents_tokenized = [word_tokenize(self.remove_stopwords(text)) for text in interim_processed_documents_whole]

            gold_kp_list = []
            for text in unique_human_skills:
                processed_text = self.preprocess(text)
                gold_kp_list.append([processed_text])
            self.document_labels = gold_kp_list

            oli_df = pd.DataFrame()
            oli_df['whole_document'] = self.original_documents_whole
            oli_df['processed_document_whole'] = self.processed_documents_whole
            oli_df['processed_document_tokenized'] = self.processed_documents_tokenized
            oli_df['keyphrases'] = gold_kp_list

            self.data_df = oli_df

        if verbose:
            print('Dataset Retrieval is complete.')

    def summarize_dataset(self, verbose):
        self.num_docs = len(self.original_documents_whole)

        num_tokens = 0
        for text in [word_tokenize(text) for text in self.original_documents_whole]:
            num_tokens += len(text)
        self.tokens_per_doc = num_tokens / self.num_docs

        num_gold_kp = 0
        num_kp_with_stopwords = 0
        num_kp_missing_words = 0

        for index in range(0, len(self.document_labels)):
            gold_kps = self.document_labels[index]
            num_gold_kp += len(gold_kps)
            for gold_kp in gold_kps:
                tokenized_gold_kp = word_tokenize(gold_kp)
                kp_has_stopwords = False
                keyphrase_missing_words = False
                for token in tokenized_gold_kp:
                    if token in self.stop_words:
                        kp_has_stopwords = True
                    if token not in self.processed_documents_whole[index]:
                        keyphrase_missing_words = True
                if kp_has_stopwords:
                    num_kp_with_stopwords += 1
                if keyphrase_missing_words:
                    num_kp_missing_words += 1

        self.gold_kp_per_doc = num_gold_kp / self.num_docs

        self.percent_stopwords_gold_kp = (num_kp_with_stopwords / num_gold_kp) * 100

        self.percent_kp_words_missing_from_doc = (num_kp_missing_words / num_gold_kp) * 100

        if verbose:
            # print('Length of the Document Labels {}'.format(len(self.document_labels)))
            print('Number of Documents in {}: {}'.format(self.dataset_name, self.num_docs))
            print('Number of Tokens per Document in {}: {:.4f}'.format(self.dataset_name, self.tokens_per_doc))
            print('Number of Gold Keyphrases per Document in {}: {:.3f}'.format(self.dataset_name, self.gold_kp_per_doc))
            print('Percentage of Gold Keyphrases that contain stopwords in {}: {:.4f}'.format(self.dataset_name, self.percent_stopwords_gold_kp))
            print('Percentage of Gold Keyphrases that contain words missing from document in {}: {:.4f}'.format(self.dataset_name, self.percent_kp_words_missing_from_doc))
            print('Dataset Summarization is complete.')

    def get_dataset_summary(self):
        return self.num_docs, self.tokens_per_doc, self.gold_kp_per_doc, self.percent_stopwords_gold_kp, self.percent_kp_words_missing_from_doc

    @staticmethod
    def calc_partial_match_stats(predicted_keyphrase_list, gold_keyphrase_list):

        predicted_keyphrases_words = [word_tokenize(text) for text in predicted_keyphrase_list]
        predicted_keyphrases_words = [item for sublist in predicted_keyphrases_words for item in sublist]
        set_pred_kp_words = set(predicted_keyphrases_words)

        target_keyphrases_words = [word_tokenize(text) for text in gold_keyphrase_list]
        target_keyphrases_words = [item for sublist in target_keyphrases_words for item in sublist]
        set_tgt_kp_words = set(target_keyphrases_words)

        common_tokens = len(list(set_pred_kp_words.intersection(set_tgt_kp_words)))

        num_predicted_words = len(list(set_pred_kp_words))
        num_target_words = len(list(set_tgt_kp_words))

        return num_predicted_words, num_target_words, common_tokens

    @staticmethod
    def calc_exact_match_stats(predicted_keyphrase_list, gold_keyphrase_list):

        set_pred_kp_list = set(predicted_keyphrase_list)
        set_tgt_kp_list = set(gold_keyphrase_list)

        common_keyphrases = len(list(set_pred_kp_list.intersection(set_tgt_kp_list)))

        num_pred_kps = len(list(set_pred_kp_list))
        num_tgt_kps = len(list(set_tgt_kp_list))

        return num_pred_kps, num_tgt_kps, common_keyphrases

    def evaluate(self, verbose=False):
        total_pred_words = 0
        total_tgt_words = 0
        total_common_words = 0

        for index in range(0, len(self.final_keyphrases)):
            predicted_keyphrase_list = self.final_keyphrases[index]
            predicted_keyphrase_list = [self.remove_stopwords(keyphrase) for keyphrase in predicted_keyphrase_list]
            predicted_keyphrase_list = [self.stem(keyphrase) for keyphrase in predicted_keyphrase_list]

            gold_keyphrase_list = self.document_labels[index]
            gold_keyphrase_list = [self.remove_stopwords(keyphrase) for keyphrase in gold_keyphrase_list]
            gold_keyphrase_list = [self.stem(keyphrase) for keyphrase in gold_keyphrase_list]
            gold_keyphrase_list = [re.sub(r"[!.?]", "", keyphrase) for keyphrase in gold_keyphrase_list]

            num_pred_words, num_tgt_words, num_common_words = self.calc_partial_match_stats(predicted_keyphrase_list, gold_keyphrase_list)
            total_pred_words += num_pred_words
            total_tgt_words += num_tgt_words
            total_common_words += num_common_words

        self.overall_precision_partial = total_common_words / total_pred_words
        self.overall_recall_partial = total_common_words / total_tgt_words
        if self.overall_precision_partial + self.overall_recall_partial > 0:
            self.overall_f1_partial = 2 * (self.overall_precision_partial * self.overall_recall_partial) / (
                    self.overall_precision_partial + self.overall_recall_partial)
        else:
            self.overall_f1_partial = 0.0

        total_pred_kps = 0
        total_tgt_kps = 0
        total_common_kps = 0

        for index in range(0, len(self.final_keyphrases)):
            predicted_keyphrase_list = self.final_keyphrases[index]
            predicted_keyphrase_list = [self.stem(keyphrase) for keyphrase in predicted_keyphrase_list]
            predicted_keyphrase_list = [self.remove_stopwords(keyphrase) for keyphrase in predicted_keyphrase_list]
            gold_keyphrase_list = self.document_labels[index]
            gold_keyphrase_list = [self.stem(keyphrase) for keyphrase in gold_keyphrase_list]
            gold_keyphrase_list = [self.remove_stopwords(keyphrase) for keyphrase in gold_keyphrase_list]

            num_pred_kps, num_tgt_kps, num_common_kps = self.calc_exact_match_stats(predicted_keyphrase_list, gold_keyphrase_list)
            total_pred_kps += num_pred_kps
            total_tgt_kps += num_tgt_kps
            total_common_kps += num_common_kps

        self.overall_precision_exact = total_common_kps / total_pred_kps
        self.overall_recall_exact = total_common_kps / total_tgt_kps
        if self.overall_precision_exact + self.overall_recall_exact > 0:
            self.overall_f1_exact = 2 * (self.overall_precision_exact * self.overall_recall_exact) / (
                    self.overall_precision_exact + self.overall_recall_exact)
        else:
            self.overall_f1_exact = 0.0

        if verbose:
            print('Overall Precision, Recall, and F1 (Partial Match): {:.4f} {:.4f} {:.4f}'.format(self.overall_precision_partial, self.overall_recall_partial, self.overall_f1_partial))
            print(
                'Overall Precision, Recall, and F1 (Exact Match): {:.4f} {:.4f} {:.4f}'.format(self.overall_precision_exact,
                                                                                               self.overall_recall_exact,
                                                                                               self.overall_f1_exact))
            print('AKE algorithm evaluation is complete.')

    def run_keybert(self, top_n, verbose=False):
        kb_model = KeyBERT(model='sentence-transformers/all-MiniLM-L6-v2')
        keyphrases_list = []
        document_count = 0
        for document in self.processed_documents_whole:
            doc_ngrams = []
            for sentence in sent_tokenize(document):
                valid = self.valid_text(sentence)
                if valid:
                    cv = CountVectorizer(ngram_range=(1,3), stop_words=self.stop_words).fit([sentence])
                    sent_ngrams = cv.get_feature_names()
                    doc_ngrams += sent_ngrams
            # if document_count == 25:
                # doc_top_n = kb_model.extract_keywords(document, candidates=doc_ngrams, top_n=20)
                # print(doc_top_n)
            doc_top_n = kb_model.extract_keywords(document, candidates=doc_ngrams)  # , top_n=top_n)
            # print('\n-----')
            # print(document)
            # print(doc_top_n)
            # print('-----')
            doc_top_n = doc_top_n[0:top_n]
            doc_kps = [kp for kp, _ in doc_top_n]
            keyphrases_list.append(doc_kps)
            document_count += 1

        self.final_keyphrases = keyphrases_list

        if verbose:
            print('Example Keyphrases Extracted by KeyBERT: {}'.format(self.final_keyphrases[13]))
            print('KeyBERT keyword extraction is complete.')

    def run_TextRank(self, top_n, verbose=False):
        keyphrases_list = [keywords(text, additional_stopwords=['correct', 'true', 'false', 'yes', 'following', 'mathrm']).split('\n')[0:top_n] for text in self.processed_documents_whole]
        self.final_keyphrases = keyphrases_list

        if verbose:
            print('Example Keyphrases Extracted by TextRank: {}'.format(self.final_keyphrases[13]))
            print('TextRank keyword extraction is complete.')

    def run_SingleRank(self, top_n, verbose=False):
        keyphrases_list = []
        keyphrase_scores_list = []

        for text in self.processed_documents_whole:
            extractor = pke.unsupervised.SingleRank()
            extractor.load_document(text)
            extractor.candidate_selection()
            extractor.candidate_filtering(stoplist=self.stop_words)
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=top_n)
            kps = [x for x, y in keyphrases]
            scores = [y for x, y in keyphrases]
            keyphrases_list.append(kps)
            keyphrase_scores_list.append(scores)

        self.final_keyphrases = keyphrases_list
        self.final_keyphrase_scores = keyphrase_scores_list

        if verbose:
            print('Example Keyphrases Extracted by SingleRank: {}'.format(self.final_keyphrases[13]))
            print('SingleRank keyword extraction is complete.')

    def run_TopicRank(self, top_n, verbose=False):
        keyphrases_list = []
        keyphrase_scores_list = []

        for text in self.processed_documents_whole:
            extractor = pke.unsupervised.TopicRank()
            extractor.load_document(text, language='en')
            extractor.candidate_selection(stoplist=self.stop_words)
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=top_n)
            kps = [x for x, y in keyphrases]
            scores = [y for x, y in keyphrases]
            keyphrases_list.append(kps)
            keyphrase_scores_list.append(scores)

        self.final_keyphrases = keyphrases_list
        self.final_keyphrase_scores = keyphrase_scores_list

        if verbose:
            print('Example Keyphrases Extracted by TopicRank: {}'.format(self.final_keyphrases[13]))
            print('TopicRank keyword extraction is complete.')

    def run_MultipartiteRank(self, top_n, verbose=False):
        keyphrases_list = []
        keyphrase_scores_list = []

        for text in self.processed_documents_whole:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(text, language='en')
            extractor.candidate_selection(stoplist=self.stop_words)
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=top_n)
            kps = [x for x, y in keyphrases]
            scores = [y for x, y in keyphrases]
            keyphrases_list.append(kps)
            keyphrase_scores_list.append(scores)

        self.final_keyphrases = keyphrases_list
        self.final_keyphrase_scores = keyphrase_scores_list

        if verbose:
            print('Example Keyphrases Extracted by MultipartiteRank: {}'.format(self.final_keyphrases[13]))
            print('MultipartiteRank keyword extraction is complete.')

    def run_RAKE(self, top_n, verbose=False):
        keyphrases_list = []

        for text in self.processed_documents_whole:
            r = Rake(stopwords=self.stop_words)
            r.extract_keywords_from_text(text)
            keyphrases = r.get_ranked_phrases()[0:top_n]
            keyphrases_list.append(keyphrases)

        self.final_keyphrases = keyphrases_list

        if verbose:
            print('Example Keyphrases Extracted by RAKE: {}'.format(self.final_keyphrases[13]))
            print('RAKE keyword extraction is complete.')

    def run_YAKE(self, top_n, verbose=False):
        keyphrases_list = []
        keyphrase_scores_list = []

        for text in self.processed_documents_whole:
            kw_extractor = KeywordExtractor(lan='en', n=3, top=top_n, stopwords=self.stop_words)
            kp_scores = kw_extractor.extract_keywords(text=text)
            kps = [x for x, y in kp_scores]
            scores = [y for x, y in kp_scores]
            keyphrases_list.append(kps)
            keyphrase_scores_list.append(scores)

        self.final_keyphrases = keyphrases_list
        self.final_keyphrase_scores = keyphrase_scores_list

        if verbose:
            print('YAKE keyword extraction is complete.')
