''' Importing Libraries '''
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model


''' Loading Dataset '''
dataset = open('TrumpTweets.txt')
dataset = dataset.readlines()


''' Assigning data as X '''
X = dataset


''' Defining Stopwords '''
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])


''' Tokenizing Words '''
Max_No_Words = 20000
Max_Sent_Len = 250

raw_data_test = X

no_classes = 5

tokenizer = RegexpTokenizer(r'\w+')

print("pre-processing statuses...")
processed_data_test = []

for data in tqdm(raw_data_test):
    tokens = tokenizer.tokenize(data)
    filtered = [word for word in tokens if word not in stop_words]
    processed_data_test.append(" ".join(filtered))
    
print("tokenizing input data...")
tokenizer = Tokenizer(num_words=Max_No_Words, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_data_test)
word_seq_test = tokenizer.texts_to_sequences(processed_data_test)
word_index = tokenizer.word_index
print("Dictionary size = ", len(word_index))

#pad sequences
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=Max_Sent_Len)

print("Done !!")


''' Embedding Words '''
embed_dim = 300

print('loading and processing word embeddings...')
EMBEDDING_FILE = 'wiki-news-300d-1M.vec'

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))
nb_words = min(Max_No_Words, len(word_index))

word_index = tokenizer.word_index
embedding_matrix_crawl = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= Max_No_Words:
        continue
    
    embedding_vector = embeddings_index.get(word)
    
    if embedding_vector is not None:
        embedding_matrix_crawl[i-1] = embedding_vector


del embeddings_index
print('Loading Done !!')


''' Loading the model '''
model = load_model('personality_model.h5')


''' Predicting the data '''
pred = model.predict(word_seq_test)
print(pred)


''' Showing the predicting data into a good format '''
add = np.sum(pred, axis=0)
div = [i / 8 for i in add]

big5 = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

for val, key in zip(div, big5):
    if val > 0.5:
        print('%s = Yes' %(key))
    elif val < 0.5:
        print('%s = No' %(key))
