import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance
from typing import Tuple, Union, Optional, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from importlib import reload
import constants
import data_utils
import models
import training_and_selection
import plot_accuracies


from constants import (
    SEED,
    NUM_SPLITS,
    INITIAL_CUTOFF,
    TOP,
    TARGET,
    MODEL_ACCURACIES_PATH,
    FILTERED_MODEL_ACCURACIES_PATH,
    TIME_LIMIT,
    TIME_LIMIT_CROSS_VALIDATION,
    RASHOMON_SETS_PATH,
    INITIAL_ACCURACIES_PATH
)
from models import MODELS, HYPERPARAMETERS

from data_utils import get_dataset

X, y = get_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=(SEED+1))
BASELINE = np.sum(y == 1) / np.sum(y == 0)


torch.cuda.empty_cache()
from xgboost import XGBClassifier
from models import SVMClassifier, TabRClassifier
from training_and_selection import train_test_models, get_rashomon_sets


XGBCLASSIFIER_HYPERPARAMETERS = {
    'n_estimators': [100, 200, 400, 600],
    "max_depth": [6, 8, 10, 12],
    "min_child_weight": [1, 4],
    "eta": [0.1, 0.3, 0.6],
    "subsample": [0.5, 0.8, 1],
    # "lambda": [0, 0.2, 1, 5, 12, 25],
    # "alpha": [0, 0.2, 1, 5, 12, 25],
}

SVMCLASSIFIER_HYPERPARAMETERS = {
    "degree": [2, 3, 4, 5, 8, 10],
    "kernel": ['poly', 'rbf', 'sigmoid'],
    'C': [0.1, 0.25, 0.5, 1, 2, 4, 8, 16],
    'gamma': ['auto', 0.2, 0.5, 1, 4],
    'max_iter': [30000],
}

TABRCLASSIFIER_HYPERPARAMETERS = {
    'activation': ['SiLU', 'GELU', 'Sigmoid', 'ReLU'],
    'd_main': [128, 256],
    'd_multiplier': [1.5, 2, 4, 6],
    'dropout0': [0, 0.1, 0.25],
    'dropout1': ['dropout0'],
    'context_size': [16, 32, 64],
    'encoder_n_blocks': [2],
    'predictor_n_blocks': [2],
    'seed': [69],
    'max_epochs': [2]
}


MODELS = [XGBClassifier, SVMClassifier, TabRClassifier, ]
HYPERPARAMETERS = {
    XGBClassifier.__name__ : XGBCLASSIFIER_HYPERPARAMETERS,
    SVMClassifier.__name__ : SVMCLASSIFIER_HYPERPARAMETERS,
    TabRClassifier.__name__ : TABRCLASSIFIER_HYPERPARAMETERS
}

# results = train_test_models(
#     MODELS, 
#     # {TabRClassifier.__name__ : TABRCLASSIFIER_HYPERPARAMETERS},
#     HYPERPARAMETERS,
#     X=X_train,
#     y=y_train,
#     path='tabr_tests.csv',
#     limit=50,
# )


rashomon_sets_params = get_rashomon_sets(
    models=MODELS,
    hyperparameters=HYPERPARAMETERS,
    X=X,
    y=y,
    initial_cutoff=0.15,
    top=0.04,
    initial_time_limit=50,
    cross_validation_time_limit=60,
    initial_path='results/initial_grid_search.csv',
    cross_validation_path='results/cross_validation_results.csv',
)
pickle.dump(rashomon_sets_params, open(RASHOMON_SETS_PATH, 'wb'))