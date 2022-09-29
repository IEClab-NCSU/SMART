"""Download 'distilbert-base-nli-mean-tokens' to avoid error due to
redundant downloading in concurrent runs of SMART
"""
from sentence_transformers import SentenceTransformer
print('Downloading pre-trained BERT model')
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Download wordnet for lemmatizer
import nltk
nltk.download('wordnet')