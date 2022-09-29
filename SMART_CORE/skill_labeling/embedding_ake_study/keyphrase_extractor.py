from datasets import load_dataset
from pathlib import Path
from nltk import word_tokenize
from nltk.corpus import stopwords
from glove import Corpus, Glove
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from rake_nltk import Rake
from yake import KeywordExtractor
from ake_autoencoder_2 import make_ake_dataset, create_and_run_autoencoder_model
import pandas as pd
import numpy as np
import sys
import os
import string
import nltk
import re
import itertools
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

        # Original
        # new_text = new_text.replace('_', ' ')
        # new_text = new_text.replace('-', ' ')

        # Update #1
        # print('-----')
        # print(f'Original Text: {text}')
        new_text = re.sub(r"[^\s]*_[^\s]*", "", new_text)
        # print(f'Next Version: {new_text}')
        new_text = re.sub(r"[^\s]*-[^\s]*", "", new_text)
        # print(f'Next Version: {new_text}')
        new_text = re.sub(r"[(].*[)]", "", new_text)
        # print(f'Next Version: {new_text}')

        new_text = re.sub(r"[^a-zA-Z\s]", "", new_text)
        # print(f'Next Version: {new_text}')
        # print('------')
        # new_text = "".join([char for char in new_text if char not in string.punctuation and not char.isnumeric()])
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
            # self.data_df['processed_document_whole'] = self.data_df['processed_document_whole'].apply(lambda text: self.lemmatize(text))
            # self.data_df['processed_document_whole'] = self.data_df['processed_document_whole'].apply(lambda text: self.remove_stopwords(text))
            self.processed_documents_whole = self.data_df.processed_document_whole.values

            # remove stopwords from the text for each document. store tokenized version and string version as class
            # variables.
            # self.data_df['processed_document_tokenized'] = self.data_df['processed_document_whole'].apply(lambda text: self.remove_stopwords(text))
            self.data_df['processed_document_tokenized'] = self.data_df['processed_document_whole'].apply(lambda text: [word for word in word_tokenize(self.remove_stopwords(text))])
            self.processed_documents_tokenized = self.data_df.processed_document_tokenized.values
            # self.processed_documents_whole = [" ".join(tokenized_text) for tokenized_text in self.processed_documents_tokenized]
            # self.processed_documents_tokenized = [[word for word in text if len(word) > 2] for text in
            # self.processed_documents_tokenized]

            self.data_df['keyphrases'] = self.data_df['extractive_keyphrases'].apply(lambda x: x.tolist())  + self.data_df['abstractive_keyphrases'].apply(lambda x: x.tolist())
            self.data_df['keyphrases'] = self.data_df['keyphrases'].apply(lambda kp_list: np.array(kp_list))
            self.data_df['keyphrases'] = self.data_df['keyphrases'].apply(lambda kp_list: [self.preprocess(keyphrase) for keyphrase in kp_list])
            # self.data_df['processed_keyphrases'] = self.data_df['keyphrases'].apply(lambda kp_list: [self.lemmatize(keyphrase) for keyphrase in kp_list])
            # print(self.data_df.processed_keyphrases.values[82])
            # self.data_df['processed_keyphrases'] = self.data_df['processed_keyphrases'].apply(lambda kp_list: [self.remove_stopwords(keyphrase) for keyphrase in kp_list])

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
                # print(len(unique_human_skills))
            elif oli_labels == 'verbose':
                unique_human_skills = full_df['Full_Human_Skill_Description'].unique()
                # print(len(unique_human_skills))
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
            interim_processed_documents_whole = [re.sub(pattern, "", text) for text in interim_processed_documents_whole]
            self.original_documents_whole = interim_processed_documents_whole
            interim_processed_documents_whole = [text.lower() for text in interim_processed_documents_whole]
            interim_processed_documents_whole = [self.preprocess(text) for text in interim_processed_documents_whole]
            # interim_processed_documents_whole = [self.lemmatize(text) for text in interim_processed_documents_whole]
            # interim_processed_documents_whole = [self.remove_stopwords(text) for text in interim_processed_documents_whole]

            self.processed_documents_whole = interim_processed_documents_whole

            # remove stopwords from the text for each document. store tokenized version and string version as class
            # variables.
            self.processed_documents_tokenized = [word_tokenize(self.remove_stopwords(text)) for text in interim_processed_documents_whole]
            # self.processed_documents_whole = [" ".join(tokenized_text) for tokenized_text in self.processed_documents_tokenized]
            # self.processed_documents_tokenized = [[word for word in text if len(word) > 2] for text in self.processed_documents_tokenized]

            gold_kp_list = []
            for text in unique_human_skills:
                processed_text = self.preprocess(text)
                gold_kp_list.append([processed_text])
                # processed_text = self.lemmatize(processed_text)
                # processed_text = self.remove_stopwords(processed_text)
                # label_list.append([processed_text])
            self.document_labels = gold_kp_list

            oli_df = pd.DataFrame()
            oli_df['whole_document'] = self.original_documents_whole
            oli_df['processed_document_whole'] = self.processed_documents_whole
            oli_df['processed_document_tokenized'] = self.processed_documents_tokenized
            oli_df['keyphrases'] = gold_kp_list

            self.data_df = oli_df

        if verbose:
            # print('Example Original Document (String): {}'.format(self.original_documents_whole[125]))
            # print('Example Processed Document (String): {}'.format(self.processed_documents_whole[125]))
            # print('Example Processed Document (Tokenized): {}'.format(self.processed_documents_tokenized[125]))
            # print('Example Gold Keyphrase List: {}'.format(self.document_labels[125]))
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

    def get_n_grams(self, doc_list, ngram=1):
        ngrams_list = []
        for text in doc_list:
            temp_ngrams = zip(*[text[i:] for i in range(0, ngram)])
            ngrams = [' '.join(ngram) for ngram in temp_ngrams]
            if ngram == 1:
                ngrams = np.unique(np.array([word for word in ngrams if len(word) > 2])).tolist()
                ngrams = [ngram for ngram in ngrams if ngram not in self.stop_words]
            else:
                ngrams = np.unique(np.array([word for word in ngrams if len(word) > 4])).tolist()
                ngrams = [ngram for ngram in ngrams if word_tokenize(ngram)[0] not in self.stop_words and word_tokenize(ngram)[-1] not in self.stop_words]
            ngrams_list.append(ngrams)
        return ngrams_list

    def get_candidate_keyphrases(self, cand_method, verbose=False):
        if cand_method == 'ngrams':
            tokenized = [word_tokenize(text) for text in self.processed_documents_whole]
            unigrams = self.get_n_grams(tokenized)
            bigrams = self.get_n_grams(tokenized, ngram=2)
            trigrams = self.get_n_grams(tokenized, ngram=3)

            candidates_list = []
            num_candidates = 0
            num_kp = 0
            num_kp_missing_in_cand = 0

            for index in range(0, len(self.processed_documents_whole)):
                candidates = unigrams[index] + bigrams[index] + trigrams[index]
                # if index == 0:
                    # print('Original Document: {}'.format(self.original_documents_whole[0]))
                    # print('\nProcessed Document (Whole): {}'.format(self.processed_documents_whole[0]))
                    # print('\nProcessed Document (Stopword Removal and Tokenized): {}'.format(" ".join(self.processed_documents_tokenized[0])))
                    # print('\nUnigrams, Bigrams, and Trigrams for first document: {}{}{}'.format(unigrams[0], bigrams[0], trigrams[0]))
                    # print(candidates)
                candidate_dict = {}
                candidate_dict['unigrams'] = unigrams[index]
                candidate_dict['other'] = bigrams[index] + trigrams[index]
                num_candidates += len(candidates)
                for kp in self.document_labels[index]:
                    num_kp += 1
                    if kp not in candidates:
                        num_kp_missing_in_cand += 1
                candidates_list.append(candidate_dict)

        elif cand_method == 'parsing':
            tagged_docs = [nltk.pos_tag(word_tokenize(doc)) for doc in self.processed_documents_whole]
            noun_phrase_grammar = r""" NP: {<JJ.?>*<NN>+}
                                           {<JJ.?>*<NNS>+}
                                           """
            doc_parser = nltk.RegexpParser(noun_phrase_grammar)
            results = [doc_parser.parse(document) for document in tagged_docs]
            # print(results[82])

            candidates_list = []
            num_candidates = 0
            num_kp = 0
            num_kp_missing_in_cand = 0

            for index in range(0, len(self.processed_documents_whole)):
                candidates = []
                candidate_dict = {}
                candidate_dict['unigrams'] = []
                candidate_dict['other'] = []
                result = results[index]
                for subtree in result.subtrees():
                    if subtree.label() == 'NP':
                        candidate = [w for w, t in subtree.leaves()]
                        candidate_string = " ".join(candidate)
                        if candidate_string not in candidates:
                            candidates.append(candidate_string)
                            num_candidates += 1
                            if len(candidate) == 1:
                                candidate_dict['unigrams'] += [candidate_string]
                            else:
                                candidate_dict['other'] += [candidate_string]

                for kp in self.document_labels[index]:
                    num_kp += 1
                    if kp not in candidates:
                        num_kp_missing_in_cand += 1
                candidates_list.append(candidate_dict)

        else:
            print("Invalid use of flag for candidate keyphrase extraction method.")
            sys.exit()

        self.candidate_list = candidates_list
        self.cand_per_doc = num_candidates / self.num_docs
        self.percent_kp_missing_in_cand = (num_kp_missing_in_cand / num_kp) * 100

        if verbose:
            # print('Example Candidate Keyphrase List: {}'.format(self.candidate_list[0]))
            print('Number of Candidate Keyphrases per Document ({}): {}'.format(cand_method, self.cand_per_doc))
            print('Percentage of Gold Keyphrases Not in Candidate Keyphrase List ({}): {}'.format(cand_method, self.percent_kp_missing_in_cand))
            print('Candidate Keyphrase Generation is complete.')

    @staticmethod
    def train_and_produce_glove(tokenized_text, cand_unigrams, cand_others, embed_dimension=384):
        corpus = Corpus()
        corpus.fit([tokenized_text])
        glove = Glove(no_components=embed_dimension, learning_rate=0.05, random_state=2022)
        glove.fit(corpus.matrix, epochs=50)
        glove.add_dictionary(corpus.dictionary)

        reference_vector = np.zeros(embed_dimension)
        for token in tokenized_text:
            embedding = glove.word_vectors[glove.dictionary[token]]
            reference_vector += embedding
        reference_vector = reference_vector / len(tokenized_text)
        reference_vector = reference_vector

        cand_uni_embeddings = []
        for token in cand_unigrams:
            if token in glove.dictionary:
                embedding = glove.word_vectors[glove.dictionary[token]]
            else:
                embedding = np.zeros(embed_dimension)
            cand_uni_embeddings.append(embedding)

        cand_other_embeddings = []
        for candidate in cand_others:
            cand_vector = np.zeros(embed_dimension)
            for token in candidate:
                if token in glove.dictionary:
                    token_embedding = glove.word_vectors[glove.dictionary[token]]
                    cand_vector += token_embedding
            cand_other_embeddings.append(cand_vector)

        return reference_vector, cand_uni_embeddings, cand_other_embeddings

    @staticmethod
    def train_and_produce_w2v(w2v_model, tokenized_text, embed_dimension=384):
        w2v_embedding = np.zeros(embed_dimension)
        for token in tokenized_text:
            if token in w2v_model.wv:
                token_embedding = w2v_model.wv.word_vec(token)
                w2v_embedding += token_embedding
        w2v_embedding = w2v_embedding / len(tokenized_text)

        return np.array(w2v_embedding)

    def get_embeddings(self, embedding_method, verbose=False):
        embedding_dimension = 384

        candidate_unigram_list = [candidates_dict['unigrams'] for candidates_dict in self.candidate_list]
        candidate_other_list = [candidates_dict['other'] for candidates_dict in self.candidate_list]

        if embedding_method == 'w2v':
            w2v_model = Word2Vec(sentences=self.processed_documents_tokenized, vector_size=embedding_dimension, window=5, min_count=5, epochs=100, seed=2022, workers=1)
            doc_embeddings = [[self.train_and_produce_w2v(w2v_model, tokenized_text)] for tokenized_text in self.processed_documents_tokenized]
            cand_unigram_embeddings = []
            cand_other_embeddings = []
            for candidates in candidate_unigram_list:
                doc_unigram_cand_embeddings = [self.train_and_produce_w2v(w2v_model, word_tokenize(text)) for text in candidates]
                cand_unigram_embeddings.append(doc_unigram_cand_embeddings)
            for candidates in candidate_other_list:
                doc_other_cand_embeddings = [self.train_and_produce_w2v(w2v_model, word_tokenize(text)) for text in candidates]
                cand_other_embeddings.append(doc_other_cand_embeddings)

        elif embedding_method == 'glove':

            doc_embeddings = []
            cand_unigram_embeddings = []
            cand_other_embeddings = []

            for index in range(0, len(self.processed_documents_tokenized)):
                doc = self.processed_documents_tokenized[index]
                cand_unigrams = candidate_unigram_list[index]
                cand_others = candidate_other_list[index]

                document_embedding, cand_uni_embed, cand_other_embed = self.train_and_produce_glove(doc, cand_unigrams, cand_others)
                doc_embeddings.append(document_embedding)
                cand_unigram_embeddings.append(cand_uni_embed)
                cand_other_embeddings.append(cand_other_embed)

        elif embedding_method == 'd2v':

            documents = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(self.processed_documents_tokenized)]
            d2v_model = Doc2Vec(documents, vector_size=embedding_dimension, min_count=1, epochs=100, seed=2022, workers=1)
            doc_embeddings = [np.array(d2v_model.infer_vector(tokenized_text, steps=50)).reshape(1, -1) for tokenized_text in self.processed_documents_tokenized]
            cand_unigram_embeddings = []
            cand_other_embeddings = []
            for candidates in candidate_unigram_list:
                doc_unigram_cand_embeddings = [d2v_model.infer_vector(word_tokenize(text), steps=50) for text in candidates]
                cand_unigram_embeddings.append(doc_unigram_cand_embeddings)
            for candidates in candidate_other_list:
                doc_other_cand_embeddings = [d2v_model.infer_vector(word_tokenize(text), steps=50) for text in candidates]
                cand_other_embeddings.append(doc_other_cand_embeddings)

        elif embedding_method == 'bert-cls':
            sys.exit()
        elif embedding_method == 'bert-sum-last4':
            sys.exit()
        elif embedding_method == 'bert-concat-last4':
            sys.exit()
        elif embedding_method == 'sbert-mean-pooling':
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            doc_embeddings = [np.array(model.encode(" ".join(text))).reshape(1, -1) for text in self.processed_documents_tokenized]
            cand_unigram_embeddings = []
            cand_other_embeddings = []
            for candidates in candidate_unigram_list:
                doc_unigram_cand_embeddings = [model.encode(text) for text in candidates]
                cand_unigram_embeddings.append(doc_unigram_cand_embeddings)
            for candidates in candidate_other_list:
                doc_other_cand_embeddings = [model.encode(text) for text in candidates]
                cand_other_embeddings.append(doc_other_cand_embeddings)
            del model
        else:
            print("Invalid use of flag for embedding method.")
            sys.exit()

        self.document_embeddings = doc_embeddings
        self.unigram_candidate_embeddings = cand_unigram_embeddings
        self.other_candidate_embeddings = cand_other_embeddings

        if verbose:
            # print('# Documents : # Document Embeddings {}:{}'.format(self.num_docs, len(self.document_embeddings)))
            # print('# Documents : # Unigram Candidate Embedding Lists {}:{}'.format(self.num_docs, len(self.unigram_candidate_embeddings)))
            # print('# Documents : # Other Candidate Embedding Lists {}:{}'.format(self.num_docs, len(self.other_candidate_embeddings)))
            # print('Example # Candidate Keyphrases : Example # Candidate Keyphrase Embeddings {}:{}'.format(len(self.candidate_list[23]['unigrams']) + len(self.candidate_list[23]['other']), len(self.unigram_candidate_embeddings[23]) + len(self.other_candidate_embeddings[23])))
            print('Document Embedding is complete.')

    def get_latent_representation(self, filepath, verbose=False):
        # train autoencoder on all doc_embeddings and retrieve latent representation for documents and candidates
        combined_embeds, doc_embeds, cand_embed, cand_embed_uni, cand_embed_other = make_ake_dataset(self.document_embeddings, self.unigram_candidate_embeddings, self.other_candidate_embeddings)
        # print(doc_embeds.shape)
        # noise = np.random.normal(loc=0.5, scale=0.5, size=doc_embeds.shape)
        # noisy_doc_embeds = np.clip(doc_embeds + noise, 0, 1)
        # noisy_doc_embeds = noisy_doc_embeds.float()
        latent_doc, latent_cand_uni, latent_cand_other = create_and_run_autoencoder_model(doc_embeds, doc_embeds, cand_embed, cand_embed_uni, cand_embed_other, filepath)

        self.document_embeddings = latent_doc
        self.unigram_candidate_embeddings = latent_cand_uni
        self.other_candidate_embeddings = latent_cand_other

        if verbose:
            print('Extraction of latent representation is complete.')

    @staticmethod
    def get_candidate_scores(candidates, candidate_embeddings, document_embedding):
        scores = []
        for candidate_embedding in candidate_embeddings:
            # print('Candidate Embedding Length:Document Embedding Length - {}:{}'.format(len([candidate_embedding]), len([document_embedding])))
            # print('Candidate Embedding Shape:Document Embedding Shape - {}:{}'.format(np.array(candidate_embedding).shape, np.array(document_embedding).shape))
            cand_embed = np.array(candidate_embedding).reshape(1, -1)
            doc_embed = np.array(document_embedding).reshape(1, -1)
            # print('New Candidate Embedding Shape:New Document Embedding Shape - {}:{}'.format(cand_embed.shape, doc_embed.shape))
            score = cosine_similarity(cand_embed, doc_embed)[0][0]
            scores.append(score)

        cand_score_dict = {candidates[i]: scores[i] for i in range(len(candidates))}
        return cand_score_dict

    @staticmethod
    def get_other_cand_scores_sum(unigram_score_dict, other_candidates):
        scores = []
        for candidate in other_candidates:
            cand_score = 0
            for word in word_tokenize(candidate):
                if word in unigram_score_dict.keys():
                    cand_score += unigram_score_dict[word]
                else:
                    pass
                    # embed the word
                    # score the word
                    # add score to cand_score
            scores.append(cand_score)

        other_cand_score_dict = {other_candidates[i]: scores[i] for i in range(len(other_candidates))}
        return other_cand_score_dict

    def rank_candidates(self, ranking_method, verbose=False):
        if ranking_method == 'cos-doc-cand':
            candidates_list = [candidates_dict['unigrams'] + candidates_dict['other'] for candidates_dict in
                               self.candidate_list]
            candidate_embeddings_list = [unigram_embeds + other_embeds for unigram_embeds, other_embeds in zip(self.unigram_candidate_embeddings, self.other_candidate_embeddings)]
            scores = [self.get_candidate_scores(candidates_list[i], candidate_embeddings_list[i], self.document_embeddings[i]) for i in range(len(candidate_embeddings_list))]

            self.score_dicts = scores

        elif ranking_method == 'cos-then-sum':
            unigram_candidates_list = [candidates_dict['unigrams'] for candidates_dict in self.candidate_list]
            other_candidates_list = [candidates_dict['other'] for candidates_dict in self.candidate_list]

            unigram_scores = [self.get_candidate_scores(unigram_candidates_list[i], self.unigram_candidate_embeddings[i], self.document_embeddings[i]) for i in range(len(unigram_candidates_list))]
            other_scores = [self.get_other_cand_scores_sum(uni_score_dict, other_cands) for uni_score_dict, other_cands in zip(unigram_scores, other_candidates_list)]

            self.score_dicts = [{**uni_scores, **o_scores} for uni_scores, o_scores in zip(unigram_scores, other_scores)]

            # unigram_score_dicts = [self.get_candidate_scores(candidates_list[i], self.)]
        else:
            print("Invalid use of flag for candidate keyphrase ranking method.")
            sys.exit()

        if verbose:
            # print('Example # Scores for a Document: {}'.format(len(list(self.score_dicts[23].keys()))))
            # print('Example Scores for a Document: {}'.format(self.score_dicts[23]))
            print('Number of Score Dictionaries: {}'.format(len(self.score_dicts)))
            print('Candidate Keyphrase Ranking is complete.')

    @staticmethod
    def sort_and_select_keyphrase(score_dict, n):
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_scores = dict(sorted_scores)
        keyphrases = list(sorted_scores.keys())[:n]
        keyphrase_scores = list(sorted_scores.values())[:n]
        keyphrase_scores = [item for item in keyphrase_scores]
        # print('Top-N Keyphrases and Scores: {} / {}'.format(keyphrases, keyphrase_scores))

        return keyphrases, keyphrase_scores

    @staticmethod
    def max_sum_similarity(doc_embedding, word_embeddings, words, top_n, nr_candidates):
        # (self, doc_embedding: np.ndarray, word_embeddings: np.ndarray, words: List[str], top_n: int,
        # nr_candidates: int):  #-> List[Tuple[str, float]]:

        """ Calculate Max Sum Distance for extraction of keywords

        We take the 2 x top_n most similar words/phrases to the document.
        Then, we take all top_n combinations from the 2 x top_n words and
        extract the combination that are the least similar to each other
        by cosine similarity.

        NOTE:
            This is O(n choose top_n) and therefore not advised if you use a large top_n

        Arguments:
            doc_embedding: The document embeddings
            word_embeddings: The embeddings of the selected candidate keywords/phrases
            words: The selected candidate keywords/keyphrases
            top_n: The number of keywords/keyphrases to return
            nr_candidates: The number of candidates to consider

        Returns:
             List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
        """
        if len(words) < nr_candidates:
            nr_candidates = len(words)
        if nr_candidates < top_n:
            top_n = nr_candidates

        # Calculate distances and extract keywords
        distances = cosine_similarity(doc_embedding, word_embeddings)
        # print('MSS Distances: {}'.format(distances))
        # print('MSS Sorted Distances: {}'.format(list(distances.argsort()[0][:])))
        distances_words = cosine_similarity(word_embeddings, word_embeddings)
        # print('MSS Distances between Words: {}'.format(distances_words))

        # Get 2*top_n words as candidates based on cosine similarity
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        # print('MSS Word Indices: {}'.format(words_idx))
        words_vals = [words[index] for index in words_idx]
        # print('MSS Words: {}'.format(words_vals))
        candidates = distances_words[np.ix_(words_idx, words_idx)]

        # Calculate the combination of words that are the least similar to each other
        min_sim = 100_000
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([candidates[i][j] for i in combination for j in combination if i != j])
            if sim <= min_sim:
                # print('Newly Selected MMS Candidate: {}'.format(combination))
                candidate = combination
                min_sim = sim

        return [(words_vals[idx], round(float(distances[0][idx]), 4)) for idx in candidate]

    @staticmethod
    def mmr(doc_embedding, word_embeddings, words, top_n, diversity):
        # doc_embedding: np.ndarray, word_embeddings: np.ndarray, words: List[str], top_n: int = 5,
        # diversity: float = 0.8):  # -> List[Tuple[str, float]]:

        """ Calculate Maximal Marginal Relevance (MMR)
        between candidate keywords and the document.


        MMR considers the similarity of keywords/keyphrases with the
        document, along with the similarity of already selected
        keywords and keyphrases. This results in a selection of keywords
        that maximize their within diversity with respect to the document.

        Arguments:
            doc_embedding: The document embeddings
            word_embeddings: The embeddings of the selected candidate keywords/phrases
            words: The selected candidate keywords/keyphrases
            top_n: The number of keywords/keyphrases to return
            diversity: How diverse the select keywords/keyphrases are.
                       Values between 0 and 1 with 0 being not diverse at all
                       and 1 being most diverse.

        Returns:
             List[Tuple[str, float]]: The selected keywords/keyphrases with their distances

        """

        # Extract similarity within words, and between words and the document
        word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
        # print('MMR Candidate-Document Similarity: {}'.format(word_doc_similarity))
        word_similarity = cosine_similarity(word_embeddings)
        # print('MMR Word Embedding Similarity: {}'.format(word_similarity))

        # Initialize candidates and already choose best keyword/keyphrase
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        if top_n > len(words):
            size = len(words)
        else:
            size = top_n

        for _ in range(size - 1):
            # Extract similarities within candidates and
            # between candidates and selected keywords/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            # print('MMR Candidate Similarities: {}'.format(candidate_similarities))
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
            # print('MMR Target Similarities: {}'.format(target_similarities))

            # print('\n\nStarting now:')
            # print(candidate_similarities)
            # print(target_similarities)

            # Calculate MMR
            mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return [(words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]

    def select_keyphrase(self, selection_method, n, verbose=False):
        if selection_method == 'top-n':
            for score_dict in self.score_dicts:
                keyphrases, keyphrase_scores = self.sort_and_select_keyphrase(score_dict, n)
                self.final_keyphrases.append(keyphrases)
                self.final_keyphrase_scores.append(keyphrase_scores)

        elif selection_method == 'top-n-mmr':
            candidates_list = [candidates_dict['unigrams'] + candidates_dict['other'] for candidates_dict in
                               self.candidate_list]
            candidate_embeddings_list = [unigram_embeds + other_embeds for unigram_embeds, other_embeds in
                                         zip(self.unigram_candidate_embeddings, self.other_candidate_embeddings)]

            for index in range(0, len(self.document_embeddings)):
                doc_embed = np.array(self.document_embeddings[index]).reshape(1, -1)
                # print(f'Shape of Document Embedding: {np.array(doc_embed).shape}')
                cand_embeds = candidate_embeddings_list[index]
                # print(f'Shape of Candidate Embeddings: {np.array(cand_embeds).shape}')
                cand_kps = candidates_list[index]
                # print(f'Candidate Keyphrases: {cand_kps}')
                # print(f'Index: {index}\n\n')
                # print(np.array(doc_embed).shape, np.array(cand_embeds).shape, np.array(cand_kps).shape)
                kp_tuples = self.mmr(doc_embed, cand_embeds, cand_kps, n, 0.7)
                kps, kp_scores = map(list, zip(*kp_tuples))
                self.final_keyphrases.append(kps)
                self.final_keyphrase_scores.append(kp_scores)

        elif selection_method == 'top-n-mss':
            candidates_list = [candidates_dict['unigrams'] + candidates_dict['other'] for candidates_dict in
                               self.candidate_list]
            candidate_embeddings_list = [unigram_embeds + other_embeds for unigram_embeds, other_embeds in
                                         zip(self.unigram_candidate_embeddings, self.other_candidate_embeddings)]

            for index in range(0, len(self.document_embeddings)):
                doc_embed = np.array(self.document_embeddings[index]).reshape(1, -1)
                cand_embeds = candidate_embeddings_list[index]
                cand_kps = candidates_list[index]
                # print(cand_kps)
                kp_tuples = self.max_sum_similarity(doc_embed, cand_embeds, cand_kps, n, 2*n)
                kps, kp_scores = map(list, zip(*kp_tuples))
                self.final_keyphrases.append(kps)
                self.final_keyphrase_scores.append(kp_scores)

        elif selection_method == 'variable':
            pass
        elif selection_method == 'variable-mmr':
            pass
        elif selection_method == 'variable-mss':
            pass
        else:
            print("Invalid use of flag for keyphrase selection method.")
            sys.exit()

        if verbose:
            # print('# Documents : # Final Keyphrase Lists {}:{}'.format(self.num_docs, len(self.final_keyphrases)))
            # print('N:Example # Final Keyphrases - {}:{}'.format(n, len(self.final_keyphrases[23])))
            # print('Example Final Keyphrase List: {}'.format(self.final_keyphrases[23]))
            # print('Example Final Keyphrase List: {}'.format(self.final_keyphrase_scores[23]))
            print('Keyphrase Selection is complete.')

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
            predicted_keyphrase_list = [self.stem(keyphrase) for keyphrase in predicted_keyphrase_list]
            predicted_keyphrase_list = [self.remove_stopwords(keyphrase) for keyphrase in predicted_keyphrase_list]
            gold_keyphrase_list = self.document_labels[index]
            gold_keyphrase_list = [self.stem(keyphrase) for keyphrase in gold_keyphrase_list]
            gold_keyphrase_list = [self.remove_stopwords(keyphrase) for keyphrase in gold_keyphrase_list]

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
