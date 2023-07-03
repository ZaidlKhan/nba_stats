import csv
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

path = "player_stats/RS_player_stats_2019.csv"

def openfile(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        data = list(reader)
        return data


def add_year_to_csv(file_path, year):
    data = openfile(file_path)

    for row in data:
        row.insert(0, year)

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(data)


def remove_first_col(file_path):
    df = pd.read_csv(file_path, delimiter=';')
    df = df.drop(df.columns[0], axis=1)
    df.to_csv(file_path, index=False, sep=';')


def replace_string(file_path, str1, str2):
    with open(file_path, 'r') as file:
        data = file.read()

    data = data.replace(str1, str2)

    with open(file_path, 'w') as file:
        file.write(data)


data = pd.read_csv(path, sep=";")

# le = preprocessing.LabelEncoder()
# year = le.fit_transform(data["YEAR"])
# full_name = le.fit_transform(data["FULL NAME"])
# team = le.fit_transform(data["TEAM"])
# position = le.fit_transform(data["POS"])
# age = le.fit_transform(data["AGE"])
# gp = le.fit_transform(data["GP"])
# mpg = le.fit_transform(data["MPG"])
# min_percent = le.fit_transform(data["MIN%"])
# usg_percent = le.fit_transform(data["USG%"])
# to_percent = le.fit_transform(data["TO%"])
# fta = le.fit_transform(data["FTA"])
# ft_percent = le.fit_transform(data["FT%"])
# two_pa = le.fit_transform(data["2PA"])
# two_p_percent = le.fit_transform(data["2P%"])
# three_pa = le.fit_transform(data["3PA"])
# three_p_percent = le.fit_transform(data["3P%"])
# efg_percent = le.fit_transform(data["eFG%"])
# ts_percent = le.fit_transform(data["TS%"])
# ppg = le.fit_transform(data["PPG"])
# rpg = le.fit_transform(data["RPG"])
# trb = le.fit_transform(data["TRB%"])
# spg = le.fit_transform(data["SPG"])
# bpg = le.fit_transform(data["BPG"])
# topg = le.fit_transform(data["TOPG"])
# vi = le.fit_transform(data["VI"])
# ortg = le.fit_transform(data["ORTG"])
# drtg = le.fit_transform(data["DRTG"])

data = data[["AGE", "GP", "MPG", "MIN%", "USG%", "TO%", "FTA",
             "FT%", "2PA", "2P%", "3PA", "3P%", "eFG%", "TS%", "PPG", "RPG", "TRB%", "APG", "AST%",
             "SPG", "BPG", "TOPG", "VI", "ORTG", "DRTG"]]

predict = "PPG"

x = np.array(data.drop([predict]), axis=1)
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)
