import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace("?", -9999, inplace=True)
df.drop(["id"], 1, inplace=True)

X = np.array(df.drop(["class"], 1))
y = np.array(df["class"])                      

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)                       

accuracy = clf.score(X_test, y_test)            
print(accuracy)

ex_data = np.array([4, 2, 1, 3, 2, 1, 3, 2, 1])
ex_data = ex_data.reshape(1, -1)

prediction = clf.predict(ex_data)
print(prediction)
