import os
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


SEED = 42
TARGET = "decision"
FEATURES = ["gender", "age", "age_o", "race", "race_o", "importance_same_race", "importance_same_religion",
          "pref_o_attractive", "pref_o_sincere", "pref_o_intelligence",
          "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests", "attractive_o", "sinsere_o", "intelligence_o", "funny_o",
          "ambitous_o", "shared_interests_o", "attractive_important", "sincere_important", "intellicence_important", "funny_important", "ambtition_important",
          "shared_interests_important", "attractive", "sincere", "intelligence", "funny", "ambition", "attractive_partner", "sincere_partner",
          "intelligence_partner", "funny_partner", "ambition_partner", "shared_interests_partner",
          "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
          "music", "shopping", "yoga",
          "interests_correlate", "expected_happy_with_sd_people", "expected_num_matches", "expected_num_interested_in_me",
          "like", "guess_prob_liked", "decision"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME_LIMIT = 20 if torch.cuda.is_available() else 60
TIME_LIMIT_CROSS_VALIDATION = 25 if torch.cuda.is_available() else 70

NUM_SPLITS = 5
INITIAL_CUTOFF = 0.75
TOP = 0.5

INITIAL_ACCURACIES_PATH = "initial_accuracies.pickle"
MODEL_ACCURACIES_PATH = "model_accuracies.csv"
FILTERED_MODEL_ACCURACIES_PATH = "filtered_model_accuracies.csv"
RASHOMON_SETS_PATH = "rashomon_sets_params.pickle"

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