import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from typing import Tuple

SEED = 42
NUM_SPLITS = 10
TARGET = "decision"
FEATURES = ['gender', 'age', 'age_o', 'race', 'race_o', 'importance_same_race', 'importance_same_religion',
          'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
          'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o',
          'ambitous_o', 'shared_interests_o', 'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important',
          'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner',
          'intelligence_partner', 'funny_partner', 'ambition_partner', 'shared_interests_partner',
          'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts',
          'music', 'shopping', 'yoga',
          'interests_correlate', 'expected_happy_with_sd_people', 'expected_num_matches', 'expected_num_interested_in_me',
          'like', 'guess_prob_liked', 'decision']


def one_hot_encode(df, features):
    for feature in features:
        dummies = pd.get_dummies(df.loc[:, feature], prefix=feature)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(feature, axis=1)
    return df


def get_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv("SpeedDating.csv", index_col=0)

    dataset = dataset.loc[:, FEATURES]
    dataset.loc[:, 'gender'] = (dataset.loc[:, 'gender'] == 'female') # one hot encode gender
    dataset = one_hot_encode(dataset, ['race', 'race_o'])
    dataset = dataset.apply(pd.to_numeric, errors='coerce', axis=1)
    dataset = dataset.fillna(dataset.mean())
    # print(dataset.head())
    X, y = dataset.loc[:, dataset.columns != TARGET], dataset.loc[:, TARGET]

    return X, y
