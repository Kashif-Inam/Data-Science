import pandas as pd
import math, datetime, arrow
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

df = pd.read_csv("GoogleStock.csv")

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100
df["PCT_Change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100

df = df[["Adj. Close", "HL_PCT", "PCT_Change", "Adj. Volume"]]

forecast_col = "Adj. Close" 
df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

df["Label"] = df[forecast_col].shift(-forecast_out) 

X = np.array(df.drop(["Label"], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out] 
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df["Label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

pd.set_option('display.max_columns', 10)
# print(df.head())
# print (accuracy)
# print(forecast_set, accuracy, forecast_out)

df["Forecast"] = np.nan 
last_date = df.iloc[-1].name 
last_unix = arrow.get(last_date).timestamp
one_day = 86400                     # 86400 sec 1 day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]
    
df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
# print(df.tail())
