import csv
import pandas as pd
import glob

path = 'stats/RS_player_stats_2023.csv'


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


