# Download 'distilbert-base-nli-mean-tokens' to avoid error due to redundant downloading in concurrent SMART runs

from sentence_transformers import SentenceTransformer
from transformers import BertModel
print('Downloading pre-trained BERT models')
model = SentenceTransformer('paraphrase-distilroberta-base-v2')
model2 = BertModel.from_pretrained('bert-base-uncased')

# Download wordnet for lemmatizer
import nltk
nltk.download('wordnet')