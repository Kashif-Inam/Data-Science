import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing, neighbors

bankdata = pd.read_csv("Dataset/bill_authentication.csv")

X = np.array(bankdata.drop('Class', 1))
y = np.array(bankdata['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print("Accuracy we got: ", score)

pred = np.array([0.88872,5.3449,2.045,-0.19355])
pred = pred.reshape(1, -1)
print("Our currency is: ", clf.predict(pred))
