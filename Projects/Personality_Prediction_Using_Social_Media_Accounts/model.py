''' Importing Libraries '''
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, GlobalMaxPooling1D
from keras.layers import LSTM, GRU, Embedding, Dropout, GlobalAveragePooling1D, concatenate, Input


''' Loading Dataset '''
dataset = pd.read_csv('mypersonality_final.csv')
dataset


''' Destributing Data into X and y '''
X = dataset[['STATUS']]
y = dataset[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]


''' Setting 'y' for 1.0 and 'n' for 0.0 '''
features = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
for feature in features:
    y[feature] = y[feature].map({'y': 1.0, 'n': 0.0}).astype(int)
    
    
''' Splitting it to the training and the testing sets '''
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size= 0.2)


''' Defining Stopwords '''
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])


''' Tokenizing Words '''
Max_No_Words = 20000
Max_Sent_Len = 250

raw_data_train = train_X['STATUS'].tolist()
raw_data_test = test_X['STATUS'].tolist()

no_classes = len(features)

tokenizer = RegexpTokenizer(r'\w+')

print("pre-processing statuses...")
processed_data_train = []
processed_data_test = []

for data in tqdm(raw_data_train):
    tokens = tokenizer.tokenize(data)
    filtered = [word for word in tokens if word not in stop_words]
    processed_data_train.append(" ".join(filtered))
for data in tqdm(raw_data_test):
    tokens = tokenizer.tokenize(data)
    filtered = [word for word in tokens if word not in stop_words]
    processed_data_test.append(" ".join(filtered))
    
print("tokenizing input data...")
tokenizer = Tokenizer(num_words=Max_No_Words, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_data_train + processed_data_test)
word_seq_train = tokenizer.texts_to_sequences(processed_data_train)
word_seq_test = tokenizer.texts_to_sequences(processed_data_test)
word_index = tokenizer.word_index
print("Dictionary size = ", len(word_index))

#pad sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=Max_Sent_Len)
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


''' Creating Model '''
inp = Input(shape= (Max_Sent_Len, ))

x = Embedding(nb_words, 300, weights=[embedding_matrix_crawl], input_length= Max_Sent_Len)(inp)

x = Bidirectional(LSTM(600, return_sequences=True, dropout=0.5, recurrent_dropout=0.3))(x)
x = GlobalMaxPooling1D()(x)
x = Dense(600, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(5, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

model.compile(loss= 'binary_crossentropy', optimizer= keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
model.summary()


''' Running the model '''
model.fit(word_seq_train, train_y, epochs= 6, batch_size= 100, validation_split=0.2, verbose= 2)
