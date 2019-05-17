''' Importing Libraries '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


''' Loading and preprocessing the dataset '''
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
trainig_set = dataset_train.iloc[:, 1:2].values


''' Normalizing the data through scaling '''
sc = MinMaxScaler(feature_range= (0, 1))
training_set_scaled = sc.fit_transform(trainig_set)


''' Creating the dataset with 60 timestamp and 1 output '''
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


''' Reshaping '''
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


''' Initializing the RNN '''
regressor = Sequential()


''' Adding 1st LSTM layer and dropout of that layer '''
regressor.add(LSTM(units= 100, return_sequences= True, input_shape= (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

''' Adding 2nd LSTM layer and dropout of that layer '''
regressor.add(LSTM(units= 100, return_sequences= True))
regressor.add(Dropout(0.2))

''' Adding 3rd LSTM layer and dropout of that layer '''
regressor.add(LSTM(units= 100, return_sequences= True))
regressor.add(Dropout(0.2))

''' Adding 4th LSTM layer and dropout of that layer '''
regressor.add(LSTM(units= 100, return_sequences= True))
regressor.add(Dropout(0.2))

''' Adding 5th LSTM layer and droout of that layer '''
regressor.add(LSTM(units= 100, return_sequences= True))
regressor.add(Dropout(0.2))

''' Adding 6th LSTM layer and dropout of that layer '''
regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.2))


''' Adding the output layer '''
# yahan oe units dimension he output ki k 1 hi dimention ki cheez predict krni he.
regressor.add(Dense(units= 1))


''' Compiling the RNN '''
regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')


''' Fitting the RNN '''
regressor.fit(X_train, y_train, epochs= 150, batch_size= 32)


''' Testing the data by getting the latest stock prices '''
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
new_stock_price = dataset_test.iloc[:, 1:2].values


''' Finding the predicted stock price of january 2017 '''
dataset_total = pd.concat( (dataset_train['Open'], dataset_test['Open']), axis= 0 )
inputs = dataset_total[ len(dataset_total) - len(dataset_test) - 60: ].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

''' Reshaping '''
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


''' Predicting '''
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)


plt.plot(new_stock_price, color= 'blue', label= 'Previous Google Stock Prices')
plt.plot(predicted_stock_price, color= 'green', label= 'New Google Stock Prices')
plt.title('Google Stock Price Predictions')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
