''' Importing Liberaries '''
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


''' Loading the dataset '''
data = pd.read_csv('Tweets.csv')
data = data.sample(frac= 1).reset_index(drop= True)
print (data.shape)
print(data.head())


''' Removing un necessary columns '''
data = data[['airline_sentiment', 'text']]
print(data.head())


''' Data exploration '''
data['airline_sentiment'].value_counts().sort_index().plot.bar()


''' Data length '''
data['text'].str.len().plot.hist()


''' Data prepocessing '''
data['text'] = data['text'].str.replace('@VirginAmerica', '')
print(data.head())

data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))
print(data['text'].head())

tokenizer = Tokenizer(num_words= 5000, split=' ')
tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


''' Initializing the model '''
model = Sequential()
model.add(Embedding(5000, 256, input_length= X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences= True, dropout= 0.3, recurrent_dropout= 0.2))
model.add(LSTM(256, dropout= 0.3, recurrent_dropout= 0.2))
model.add(Dense(3, activation= 'softmax'))


''' Compiling the model '''
model.compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
model.summary()


''' Label encoding '''
y= pd.get_dummies(data['airline_sentiment']).values
[print(data['airline_sentiment'][i], y[i]) for i in range(0, 5)]


''' Splitting the data '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


''' Running the model '''
model.fit(X_train, y_train, epochs= 8, batch_size= 32, verbose=2)

model.save('sentiment_analysis.h5')


''' Making predictions on testing data '''
predictions = model.predict(X_test)
[print(data['text'][i], predictions[i], y_test[i]) for i in range(0, 5)]


''' Making prediction on a single tweet '''
twt = ['I have observed a bad gesture on your face']

twt = tokenizer.texts_to_sequences(twt)

twt = pad_sequences(twt, maxlen=33, dtype='int32', value=0)
sentiment = model.predict(twt, batch_size=1, verbose = 2)[0]

if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")
elif (np.argmax(sentiment) == 2):
    print("normal")
