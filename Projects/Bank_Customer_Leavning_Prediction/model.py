import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

""" CODE TO HANDLE NON-NUMERICAL DATA FROM A DATA-SET """
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) 
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


""" SPLITTING DATA INTO TRAINING AND TESTING """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


""" SCALING DATA """
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


""" INITIALIZING MODEL """
classifier = Sequential()


""" INITIALIZING INPUT LAYER AND 1st HIDDEN LAYER """

classifier.add(Dense(output_dim = 6, init = 'uniform', activation= 'relu', input_dim= 11))


""" INITIALIZING 2nd HIDDEN LAYER """
classifier.add(Dense(output_dim = 6, init = 'uniform', activation= 'relu'))


""" INITIALIZING OUTPUT LAYER """
classifier.add(Dense(output_dim = 1, init = 'uniform', activation= 'sigmoid'))


""" COMPILING CLASSIFIER """
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])


""" RUNNING THE MODEL """
# no. of batch_size ka bhi koi rule nai or no. of epochs hmari marzi
classifier.fit(X_train, y_train, batch_size= 50, nb_epoch= 100)


""" Predicting """
#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
#print(y_pred)

specific = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
sc.transform(specific)
y_pred = classifier.predict(specific)
y_pred = (y_pred > 0.5)
print(y_pred)
