import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pd.read_csv("Dataset/balance-scale.data.txt")


# converting strings/characters into integers
def converting_class():
    column = file["class"]
    dic = {}
    i = 0
    for f in column:
        if f not in dic:
            dic[f] = i
            i += 1
        else:
            column.replace(f, dic[f], inplace=True)
    file["class"] = column
    return dic


X = np.array(df.drop("class", 1))
y = np.array(df["class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVR()
clf.fit(X_train, y_train)
accuarcy = clf.score(X_test, y_test)

y_pred = np.array([1, 1, 4, 5])
y_pred = y_pred.reshape(1, -1)
predict = clf.predict(y_pred)

print("Prediction: ", predict)

print("Accuracy: ", accuarcy)
