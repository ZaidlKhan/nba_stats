import csv
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing, ensemble
from sklearn.neighbors import KNeighborsClassifier

path = "player_stats/NBA_Player_Stats.csv"

data = pd.read_csv(path)

data = data[["Player", "Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA",
             "2P", "2PA" , "2P%", "eFG%", "FT", "FT%", "ORB", "DRB", "TRB", "AST",
             "STL", "BLK", "TOV", "PF", "PTS"]]

data.fillna(0, inplace = True)

lebron_james_stats_only = data[data['Player'] == 'LeBron James']

print(lebron_james_stats_only)