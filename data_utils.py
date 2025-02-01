import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from typing import Tuple

from constants import FEATURES, TARGET



def one_hot_encode(df, features):
    for feature in features:
        dummies = pd.get_dummies(df.loc[:, feature], prefix=feature)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(feature, axis=1)
    return df


def get_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv("SpeedDating.csv", index_col=0)

    dataset = dataset.loc[:, FEATURES]
    dataset.loc[:, "gender"] = (dataset.loc[:, "gender"] == "female") # one hot encode gender
    dataset = one_hot_encode(dataset, ["race", "race_o"])
    dataset = dataset.apply(pd.to_numeric, errors="coerce", axis=1)
    dataset = dataset.fillna(dataset.mean())
    # print(dataset.head())
    X, y = dataset.loc[:, dataset.columns != TARGET], dataset.loc[:, TARGET]

    return X, y
