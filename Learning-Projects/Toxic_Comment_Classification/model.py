''' Importing Liberaries '''
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import codecs
from tqdm import tqdm

from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, GlobalMaxPooling1D
from keras.layers import LSTM, GRU, Embedding, Dropout, GlobalAveragePooling1D, concatenate, Input


''' Loading Dataset '''
train = pd.read_csv('Datasets/train.csv')
train.dropna(inplace=True)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train[list_classes]

# test data
test = pd.read_csv('Datasets/test.csv')
test.dropna(inplace=True)


''' Removing Stopwords '''
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])


''' Tokenizing Words '''
MAX_NB_WORDS = 100000
max_seq_len = 250

raw_docs_train = train['comment_text'].tolist()
raw_docs_test = test['comment_text'].tolist()

num_classes = len(list_classes)

tokenizer = RegexpTokenizer(r'\w+')

print("pre-processing train data...")
processed_docs_train = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))

print("pre-processing test data...")
processed_docs_test = []
for doc in tqdm(raw_docs_test):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_test.append(" ".join(filtered))


print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))

#pad sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

print("Done !!")


''' Embedding Words '''
embed_dim = 300

print('loading word embeddings...')
EMBEDDING_FILE = 'wiki-news-300d-1M.vec'

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))
nb_words = min(MAX_NB_WORDS, len(word_index))

word_index = tokenizer.word_index
embedding_matrix_crawl = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_crawl[i] = embedding_vector


del embeddings_index
print('Loading Done !!')


''' Creating Model '''
inp = Input(shape=(max_seq_len, ))

emb_crawl = Embedding(nb_words, 300, weights=[embedding_matrix_crawl], input_length = max_seq_len, trainable=False)(inp)

x = Bidirectional(LSTM(400, return_sequences=True))(emb_crawl)
x = Bidirectional(GRU (400, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
out = concatenate([avg_pool, max_pool])

out = Dense(200, activation="relu")(out)
out = Dense(y_train.shape[1], activation="sigmoid")(out)

model = Model(inputs=inp, outputs=out)

model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
print(model.summary())


''' Running the model '''
model.fit(word_seq_train, y_train, epochs=6, batch_size=80, shuffle=True, validation_split=0.1,verbose = 2)


''' Predicting testing data '''
y_test = model.predict(word_seq_test)
print(y_test)
