''' Importing liberaries '''

import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers


'''Loading the dataset'''
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)


''' Checking the no. of words in the X_train '''
print(max([max(sequence) for sequence in X_train]))


''' Decoding back the words '''
word_index = imdb.get_word_index()

# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in X_train[0]])

print(decoded_review)


''' Creating words into vectors '''
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
X_train = vectorize_sequences(X_train)
# Our vectorized test data
X_test = vectorize_sequences(X_test)


''' Defining y_train and y_test '''
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')


''' Building the neural network '''
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


''' Compiling the model '''
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


''' Fitting the model '''
model.fit(X_train, y_train, epochs=40, batch_size=512)


''' Evaluating the model '''
print(model.evaluate(X_test, y_test))


''' Predicting a single review '''
from nltk import word_tokenize
from keras.preprocessing import sequence

review= 'The movie was not so good'

word2index = imdb.get_word_index()
test=[]
for word in word_tokenize(str.lower(review)):
     test.append(word2index[word])

test=sequence.pad_sequences([test],maxlen= 10000)
sentiment = model.predict(test)

if sentiment == 0:
    print('This is a negative review')
elif sentiment == 1:
    print('This is a positive review')
