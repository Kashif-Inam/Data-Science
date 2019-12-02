''' Importing Liberaries '''

import pandas as pd
import numpy as np
from tqdm import tqdm

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Flatten, Dense, Dropout, MaxPool2D

from sklearn.model_selection import train_test_split


train = pd.read_csv("C:\\Users\\Kashif\\PycharmProjects\\FYP\\DrawsinessDetection\\Shape Predictor\\facial-keypoints-detection\\training\\training.csv")
print(train.head().T)


''' Checking the null values for each '''

train.isnull().sum()


''' Filling the null values '''

train.fillna(method = 'ffill',inplace = True)


''' Data Preprocessing '''

X = train.Image.values
del train['Image']
Y = train.values

x = []
for i in tqdm(X):
    q = [int(j) for j in i.split()]
    x.append(q)

x = np.array(x)
x = x.reshape(7049, 96,96,1)
x  = x/255.0


''' Splitting data into training and testing set '''

x_train, x_test, y_train, y_test = train_test_split(x, Y, random_state = 69, test_size = 0.1)


''' Defining the model '''

model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()


''' Training the model '''

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])
model.fit(x_train, y_train, batch_size=256, epochs=50)


''' Saving model '''

model.save('shape_predictor_68_face_landmarks.dat')
