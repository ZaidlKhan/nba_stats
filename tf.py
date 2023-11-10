import csv
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

path = "player_stats/NBA_Player_Stats.csv"

data = pd.read_csv(path)

data = data[["Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA",
             "2P", "2PA", "2P%", "eFG%", "FT", "FT%", "ORB", "DRB", "TRB", "AST",
             "STL", "BLK", "TOV", "PF", "PTS"]]

data.fillna(0, inplace=True)

predict = "PTS"

y = np.array(data[predict])
x = np.array(data.drop(predict, axis=1))
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)

imputer = SimpleImputer(strategy="mean")
imputer.fit(x_train)
x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

print(data.head())
