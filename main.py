import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing, ensemble
from sklearn.impute import SimpleImputer
import pickle
from matplotlib import style
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle

path = "player_stats/RS_player_stats_2019.csv"

data = pd.read_csv(path, sep=";")
player_test = data.iloc[318]

data = data[["AGE", "GP", "MPG", "MIN%", "USG%", "TO%", "FTA",
             "FT%", "2PA", "2P%", "3PA", "3P%", "eFG%", "TS%", "PPG", "RPG", "TRB%", "APG", "AST%",
             "SPG", "BPG", "TOPG", "VI", "ORTG", "DRTG"]]

predict = "PPG"

x = np.array(data.drop(predict, axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

imputer = SimpleImputer(strategy="mean")
imputer.fit(x_train)
x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)


# best = 0
# for num in range(30):
#    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#    imputer = SimpleImputer(strategy="mean")
#    imputer.fit(x_train)
#    x_train = imputer.transform(x_train)
#    x_test = imputer.transform(x_test)
#    linear = linear_model.LinearRegression()
#    linear.fit(x_train, y_train)
#    acc = linear.score(x_test, y_test)
#    if acc > best:
#        best = acc
#        with open("stats_model.pickle", "wb") as f:
#            pickle.dump(linear, f)

pickle_in = open("stats_model.pickle", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

