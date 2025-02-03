from functools import partial
import gc
import os
import pickle
from typing import Callable, Dict
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import lime
import lime.lime_tabular
import shap
from PyALE import ale
from tqdm import tqdm

from constants import EXPLANATIONS_PATH, SEED, SENSITIVE_FEATURES
from models import TabRClassifier
from training_and_selection import get_model, release_model_vram


def performance_metrics(model, X_test, y_test, plot=False):
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Demographic Parity": {},
        "Equal Opportunity": {},
        "Predictive Positive Rate": {},
    }

    
    for feature_name in SENSITIVE_FEATURES:
        feature_values = X_test[feature_name]
        groups = np.unique(feature_values)
        
        # Demographic parity: P(ŷ=1|A=a) for each group a
        demographic_parity = {
            group: np.mean(y_pred[feature_values == group])
            for group in groups
        }

        # Equal opportunity: P(ŷ=1|y=1, A=a)
        equal_opportunity = {
            group: np.mean(y_pred[(feature_values == group) & (y_test == 1)])
            for group in groups
        }

        predictive_positive_rate = {
            group: np.sum(y_pred[feature_values == group]) / len(y_pred[feature_values == group])
            for group in groups
        }

        metrics["Demographic Parity"][feature_name] = demographic_parity
        metrics["Equal Opportunity"][feature_name] = equal_opportunity
        metrics["Predictive Positive Rate"][feature_name] = predictive_positive_rate

    return metrics


def permutation_feature_importance(model, X_test, y_test, plot=False, gender=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # for model_class in MODEL:
        #     for model in rashomon_sets[model_class.__name__]:
        if gender is not None:
            y_test = y_test[X_test['gender'] == gender]
            X_test = X_test[X_test['gender'] == gender]
        importances = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=2,
            random_state=SEED,
            scoring="accuracy"
        )
        importances = pd.Series(importances['importances'][:, 0], index=list(X_test.columns))
        if plot:
            fig, ax = plt.subplots()
            importances.plot.bar(ax=ax)
            ax.set_title("Permutation feature importances")
            ax.set_ylabel("importance")
            fig.tight_layout()

    return importances
    

def lime_explanation(model, X_test, y_test, plot=False, sample_no=SEED):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        categorical_features = [0, 3, 4, -1]
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_test.to_numpy(), categorical_features=categorical_features, feature_names=list(X_test.columns), class_names=['negative', 'positive'])
        # for model_class in MODELS:
        #     for model in rashomon_sets[model_class.__name__]:
        explanation = lime_explainer.explain_instance(X_test.iloc[sample_no, :], model.predict_proba, num_features=5)
        if plot:
            fig = explanation.as_pyplot_figure()
            plt.plot()
    return explanation


def shap_explanation(model, X_test, y_test, plot=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # for model_class in MODELS:
        #     for model in rashomon_sets[model_class.__name__]:
        #         i += 1
        #         if i <= 4:
        #             continue
        # if hasattr(model, "feature_names_in_"):
        # model.fit(X_train.values, y_train.values)
        # print(type(X_train.iloc[:100, :]))
        predict_func = lambda x: model.predict(x)
        shap_explainer = shap.KernelExplainer(predict_func, X_test.sample(100).to_numpy(), feature_names=list(X_test.columns))
        explaination = shap_explainer(X_test.iloc[[SEED], :])
        if plot:
            shap.plots.beeswarm(explaination)
            plt.plot()

    # return explaination.values
    return explaination


def getALE(model, X_test, y_test, plot=False):
    ## 1D - continuous - with 95% CI
    features = ['funny_partner', 'expected_num_matches', 'guess_prob_liked', 'attractive_partner', 'like']
    results = {}
    for feature in features:
        ale_eff = ale(
            X=X_test, model=model, feature=[feature], grid_size=50, include_CI=True, C=0.95, plot=plot
        )
        results[feature] = ale_eff
    return results

def getALE_2D(model, X_test, y_test, plot=False, gender=0, features=['intelligence', 'intelligence_partner']):
    y_test = y_test[X_test['gender'] == gender]
    X_test = X_test[X_test['gender'] == gender]
    ale_eff = ale(
        X=X_test, model=model, feature=features, grid_size=50, include_CI=True, C=0.95, plot=plot
    )
    return ale_eff


EXPLANATION_FUNCS = {
    'lime': lime_explanation,
    'shap': shap_explanation,
    'vip': permutation_feature_importance,
    'metrics': performance_metrics,
}

EXPLANATION_FUNCS2 = {
    'lime420': partial(lime_explanation, sample_no=420),
    'lime69': partial(lime_explanation, sample_no=69),
    'ale': getALE,
    'vip_male': partial(permutation_feature_importance, gender=1),
    'vip_female': partial(permutation_feature_importance, gender=0),
}

EXPLANATION_FUNCS3 = {
    'ale0int': partial(getALE_2D, gender=0, features=['intelligence', 'intelligence_partner']),
    'ale1int': partial(getALE_2D, gender=1, features=['intelligence', 'intelligence_partner']),
    'ale0amb': partial(getALE_2D, gender=0, features=['ambition', 'ambition_partner']),
    'ale1amb': partial(getALE_2D, gender=1, features=['ambition', 'ambition_partner']),
}

def run_all_explanations(
        models,
        rashomon_sets_params,
        X_train,
        y_train,
        X_test,
        y_test,
        explanation_funcs: Dict[str, Callable] = EXPLANATION_FUNCS,
        path=EXPLANATIONS_PATH,
        plot=False
    ):

    all_explanations = []
    for model_class in models:
        results_path = f'{path}/{model_class.__name__}.pickle'
        skip_iters = 0
        ex_name = next(iter(explanation_funcs.keys()))
        if os.path.exists(results_path):
            with open(results_path, 'rb') as file:
                explanations = pickle.load(file)
            skip_iters = len(explanations[ex_name])
            print(f'Skipping {skip_iters} iterations of {model_class.__name__}')
        else:
            explanations = {name: dict() for name in explanation_funcs.keys()}

        for kwargs in tqdm(rashomon_sets_params[model_class.__name__]):
            model_idx = model_class.__name__, str(kwargs)
            if model_idx in explanations.get(ex_name, {}):
                continue
            model = get_model(model_class, kwargs, X_train, y_train)
            for name, explain_func in tqdm(explanation_funcs.items()):
                expl = explain_func(model, X_test, y_test, plot=plot)
                explanations[name][model_idx] = expl
                
            if model_class.__name__ == 'TabRClassifier':
                del model
                gc.collect()
                torch.cuda.empty_cache()
            with open(f'{path}/{model_class.__name__}.pickle', 'wb') as file:
                pickle.dump(explanations, file)
        all_explanations.append(explanations)

    return all_explanations
