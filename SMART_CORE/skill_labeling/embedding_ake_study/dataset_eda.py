from keyphrase_extractor import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil


output_path = 'dataset_eda_updated/'

if os.path.exists(output_path):
    shutil.rmtree(output_path)

os.mkdir(output_path)

keyphrase_extractor_chem_verbose = KeyphraseExtractor('oli-gen-chem', verbose=True)
for candidate_method in ['ngrams', 'parsing']:
    keyphrase_extractor_chem_verbose.get_candidate_keyphrases(candidate_method, verbose=True)
print('--------\n\n')
dataset_chem_verbose = keyphrase_extractor_chem_verbose.data_df
dataset_chem_verbose['Dataset'] = pd.Series(['oli-gen-chem' for x in range(len(dataset_chem_verbose.index))])

keyphrase_extractor_bio = KeyphraseExtractor('oli-intro-bio', verbose=True)
for candidate_method in ['ngrams', 'parsing']:
    keyphrase_extractor_bio.get_candidate_keyphrases(candidate_method, verbose=True)
print('--------\n\n')
dataset_bio = keyphrase_extractor_bio.data_df
dataset_bio['Dataset'] = pd.Series(['oli-intro-bio' for x in range(len(dataset_bio.index))])

keyphrase_extractor_inspec = KeyphraseExtractor('inspec', oli_labels='short', verbose=True)
for candidate_method in ['ngrams', 'parsing']:
    keyphrase_extractor_inspec.get_candidate_keyphrases(candidate_method, verbose=True)
print('--------\n\n')
dataset_inspec = keyphrase_extractor_inspec.data_df
dataset_inspec['Dataset'] = pd.Series(['inspec' for x in range(len(dataset_inspec.index))])
dataset_inspec = dataset_inspec.drop(['id', 'document', 'other_metadata', 'doc_bio_tags', 'extractive_keyphrases', 'abstractive_keyphrases', 'processed_document_whole', 'processed_document_tokenized'], axis=1)

keyphrase_extractor_kdd = KeyphraseExtractor('kdd', oli_labels='short', verbose=True)
for candidate_method in ['ngrams', 'parsing']:
    keyphrase_extractor_kdd.get_candidate_keyphrases(candidate_method, verbose=True)
print('--------\n\n')
dataset_kdd = keyphrase_extractor_kdd.data_df
dataset_kdd['Dataset'] = pd.Series(['kdd' for x in range(len(dataset_kdd.index))])
dataset_kdd = dataset_kdd.drop(['id', 'document', 'other_metadata', 'doc_bio_tags', 'extractive_keyphrases', 'abstractive_keyphrases', 'processed_document_whole', 'processed_document_tokenized'], axis=1)

# print(dataset_chem_verbose.head())
# print(dataset_bio.head())
# print(dataset_inspec.head())
# print(dataset_kdd.head())

# gold kp per document across all datasets
full_dataset = pd.concat([dataset_chem_verbose, dataset_bio, dataset_inspec, dataset_kdd])
full_dataset['Gold Keyphrase Count'] = full_dataset['keyphrases'].str.len().fillna(0).astype(int)

sns.boxplot(data=full_dataset, x='Dataset', y='Gold Keyphrase Count', showfliers=False)
plt.title('Number of Gold Keyphrases Per Document Across AKE Datasets')
fpath = 'dataset_eda_updated/target_keyphrases_per_document.pdf'
plt.savefig(fpath)
plt.close()

kp_list_chem = pd.Series([item[0] for item in dataset_chem_verbose['keyphrases'].values]).unique()
kp_list_bio = pd.Series([item[0] for item in dataset_bio['keyphrases'].values]).unique()

kp_list_lengths_chem = [len(word_tokenize(kp)) for kp in kp_list_chem]
kp_list_lengths_bio = [len(word_tokenize(kp)) for kp in kp_list_bio]
kp_list_lengths_inspec = [len(word_tokenize(kp)) for kp_list in dataset_inspec.keyphrases for kp in kp_list]
kp_list_lengths_kdd = [len(word_tokenize(kp)) for kp_list in dataset_kdd.keyphrases for kp in kp_list]

doc_lengths_chem = [len(word_tokenize(doc)) for doc in dataset_chem_verbose.whole_document]
doc_lengths_bio = [len(word_tokenize(doc)) for doc in dataset_bio.whole_document]
doc_lengths_inspec = [len(word_tokenize(doc)) for doc in dataset_inspec.whole_document]
doc_lengths_kdd = [len(word_tokenize(doc)) for doc in dataset_kdd.whole_document]

kp_chem = pd.DataFrame()
kp_chem['KP Lengths'] = kp_list_lengths_chem
kp_chem['Dataset'] = pd.Series(['oli-gen-chem' for x in range(len(kp_chem.index))])

doc_chem = pd.DataFrame()
doc_chem['Document Lengths'] = doc_lengths_chem
doc_chem['Dataset'] = pd.Series(['oli-gen-chem' for x in range(len(doc_chem.index))])

kp_bio = pd.DataFrame()
kp_bio['KP Lengths'] = kp_list_lengths_bio
kp_bio['Dataset'] = pd.Series(['oli-intro-bio' for x in range(len(kp_bio.index))])

doc_bio = pd.DataFrame()
doc_bio['Document Lengths'] = doc_lengths_bio
doc_bio['Dataset'] = pd.Series(['oli-intro-bio' for x in range(len(doc_bio.index))])

kp_inspec = pd.DataFrame()
kp_inspec['KP Lengths'] = kp_list_lengths_inspec
kp_inspec['Dataset'] = pd.Series(['inspec' for x in range(len(kp_inspec.index))])

doc_inspec = pd.DataFrame()
doc_inspec['Document Lengths'] = doc_lengths_inspec
doc_inspec['Dataset'] = pd.Series(['inspec' for x in range(len(doc_inspec.index))])

kp_kdd = pd.DataFrame()
kp_kdd['KP Lengths'] = kp_list_lengths_kdd
kp_kdd['Dataset'] = pd.Series(['kdd' for x in range(len(kp_kdd.index))])

doc_kdd = pd.DataFrame()
doc_kdd['Document Lengths'] = doc_lengths_kdd
doc_kdd['Dataset'] = pd.Series(['kdd' for x in range(len(doc_kdd.index))])

full_kp = pd.concat([kp_chem, kp_bio, kp_inspec, kp_kdd])
full_doc = pd.concat([doc_chem, doc_bio, doc_inspec, doc_kdd])

# length (number of tokens) per gold kp
sns.boxplot(data=full_kp, x='Dataset', y='KP Lengths', showfliers=False)
plt.title('Length of Gold Keyphrases (# Tokens) Across AKE Datasets')
fpath = 'dataset_eda_updated/target_keyphrase_length.pdf'
plt.savefig(fpath)
plt.close()

# length (number of tokens) per document
sns.boxplot(data=full_doc, x='Dataset', y='Document Lengths', showfliers=False)
plt.title('Length of Documents (# Tokens) Across AKE Datasets')
fpath = 'dataset_eda_updated/document_length.pdf'
plt.savefig(fpath)
plt.close()
