import pandas as pd

def data_clean(file_path):
    data = pd.read_csv(file_path, index_col=0)
    data = data.sort_values("date")
    data = data.reset_index(drop=True)
    del data["mp.1"]
    del data["mp_opp.1"]
    del data["index_opp"]   

    def add_target(team):
        team["target"] = team["won"].shift(-1)
        return team

    data = data.groupby("team", group_keys=False).apply(add_target)

    data.loc[pd.isnull(data["target"]), 'target'] = 2
    data["target"] = data["target"].astype(int, errors="ignore")

    nulls = pd.isnull(data)
    nulls = nulls.sum()
    nulls = nulls[nulls > 0]
    valid_columns = data.columns[~data.columns.isin(nulls.index)]
    data = data[valid_columns].copy()
    return data
