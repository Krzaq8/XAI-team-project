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

MODEL_ACCURACIES_PATH = "model_accuracies.csv"
FILTERED_MODEL_ACCURACIES_PATH = "filtered_model_accuracies.csv"

XGBCLASSIFIER_HYPERPARAMETERS = {
    "max_depth": [3, 6],
    "min_child_weight": [1, 4],
    "device": [DEVICE],
}
SVMCLASSIFIER_HYPERPARAMETERS = {
    "kernel": ["poly"],
    "degree": [1, 2],
}
TABRCLASSIFIER_HYPERPARAMETERS = {
   "d_main": [12, 36],
    "d_multiplier": [1.5, 2],
    "seed": [SEED]
}
