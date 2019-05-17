import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split

file = pd.read_csv("iris.data.txt")

# function to convert strings/characters into integers
def converting_class():
    column = file["Class"]
    dic = {}
    i = 0
    for f in column:
        if f not in dic:
            dic[f] = i
            i += 1
        else:
            column.replace(f, dic[f], inplace=True)
    file["Class"] = column
    return dic

# function to show the predicted answer with it's flower name
def prediction_convert(prediction):
    for key, val in dic.items():
        if val == prediction:
            return key


dic = converting_class()
X = np.array(file.drop("Class", 1))
y = np.array(file["Class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy:", round((accuracy * 100), 2), "%")

# predicting on a new data
data = np.array([6.2, 2.8, 4.8, 1.8])
data = data.reshape(1, -1)
prediction = clf.predict(data)

print("The flower is:", prediction_convert(prediction))
