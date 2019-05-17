import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv")

X = np.array(df.drop(['Outcome'], axis=1))
y = np.array(df["Outcome"])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

# predicting on a new data
y_pred = np.array([2, 197, 70, 45, 543, 30.5, 0.158, 53])
y_pred = y_pred.reshape(1, -1)

prediction = clf.predict(y_pred)
print("Prediction: ", prediction)

