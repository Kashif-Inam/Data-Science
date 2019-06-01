''' Importing Libraries '''
import os

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator


''' Loading dataset '''
folder_benign = "C:\\Users\\Kashif\\PycharmProjects\\DeepLearning-Course\\SkinCancerDetectionOfBenign&Melignant\\skin-cancer-malignant-vs-benign\\data\\benign"
folder_malignant = "C:\\Users\\Kashif\\PycharmProjects\\DeepLearning-Course\\SkinCancerDetectionOfBenign&Melignant\\skin-cancer-malignant-vs-benign\\data\\malignant"

# Using for reading the images to RGB format
read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# Loading pictures
ims_benign = [read(os.path.join(folder_benign, filename)) for filename in os.listdir(folder_benign)]
X_benign = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant, filename)) for filename in os.listdir(folder_malignant)]
X_malignant = np.array(ims_malignant, dtype='uint8')

# Creating labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

# Merge data and shuffle it
X = np.concatenate((X_benign, X_malignant), axis = 0)
y = np.concatenate((y_benign, y_malignant), axis = 0)
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
y = y[s]


''' Turning labels into one hot encoders '''
y = to_categorical(y, num_classes= 2)


''' Normalizing the pictures '''
X_scaled = X/255


''' Splitting the data into train and test data '''
X_train, X_test, y_train, y_test= train_test_split(X_scaled, y, test_size=0.20, random_state=42)


''' Building the model '''
input_shape = (224, 224, 3)
num_classes = 2

model = Sequential()
model.add(Conv2D(64, kernel_size= (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(128, kernel_size= (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(128, kernel_size= (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(128, kernel_size= (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss = "binary_crossentropy", metrics=["accuracy"])


''' Summery of the model '''
model.summary()


''' Running the model'''
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16)
