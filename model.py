from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import data_cleaning
import pandas as pd

data = data_cleaning.data_clean("nba_games.csv")

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="backward", cv=split)

removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = data.columns[~data.columns.isin(removed_columns)]
scaler = MinMaxScaler()
data[selected_columns] = scaler.fit_transform(data[selected_columns]) 

sfs.fit(data[selected_columns], data["target"])

predictions = list(selected_columns[sfs.get_support()])

def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []

    seasons = sorted(data["season"].unique())
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "predictions"]
        all_predictions.append(combined)
    return pd.concat(all_predictions)

prediction = backtest(data, rr, predictions)

data.groupby("home").apply(lambda x: x[x["won"] == 1].shape[0] /x.shape[0])

data_rolling = data[list(selected_columns) + ["won", "team", "season"]]

# Performance in last 10 games
def find_team_average(team):
    rolling = team.rolling(10).mean()
    return rolling

data_rolling = data_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_average)
rolling_cols = [f"{col}_5" for col in data_rolling.columns]
data_rolling.columns = rolling_cols
data = pd.concat([data, data_rolling], axis=1)
data = data.dropna()

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(data, col_name):
    return data.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

data["home_next"] = add_col(data, "home")
data["team_opp_next"] = add_col(data, "team_opp")
data["date_next"] = add_col(data, "date")

full = data.merge(data[rolling_cols + ["team_opp_next", "date_next", "team"]], 
                  left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns
selected_columns = full.columns[~full.columns.isin(removed_columns)]
sfs.fit(full[selected_columns], full["target"])

predictors = list(selected_columns[sfs.get_support()])
prediction1 = backtest(data, rf, predictions)

print(accuracy_score(prediction1["actual"], prediction1["predictions"]))