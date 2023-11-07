import numpy as np
import pandas as pd
import csv
import matplotlib as plot

path = "nba_stats/player_stats/NBA_Player_Stats.csv"

data = pd.read_csv(path, sep=";")

df = pd.DataFrame(data, {"Player": "Lebron James"})

print(df.head())
