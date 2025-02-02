from ast import literal_eval
import gc
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from itertools import product
import torch
from tqdm import tqdm
import csv
from pathlib import Path
from contextlib import contextmanager
import threading
import _thread
import signal
import pickle

from constants import (
    FILTERED_MODEL_ACCURACIES_PATH,
    MODEL_ACCURACIES_PATH,
    INITIAL_ACCURACIES_PATH,
    NUM_SPLITS,
    SEED,
    TIME_LIMIT,
    TIME_LIMIT_CROSS_VALIDATION
)


class TimeoutException(Exception):
    def __init__(self, msg=""):
        self.msg = msg


@contextmanager
def time_limit(seconds):

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
    # timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    # timer.start()
    # try:
    #     yield
    # except KeyboardInterrupt:
    #     raise TimeoutException()
    # finally:
    #     # if the action ends in specified time, timer is canceled
    #     timer.cancel()


def train_test_models(models, hyperparameters, X, y, path, limit):
    accuracies = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
    if not Path(path).is_file():
        with open(path, "w", newline="") as csvfile:
            fieldnames = ["model_class", "acc", "kwargs"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    else:
        df = pd.read_csv(path)
        for m, group in df.groupby('model_class'):
            accuracies[m] = list(zip(group['acc'].tolist(), group['kwargs'].apply(literal_eval).tolist()))
            print(m, len(accuracies[m]))
    for model_class in models:
        print(model_class.__name__)
        model_hyperparameters = hyperparameters[model_class.__name__]
        if model_class.__name__ not in accuracies:
            accuracies[model_class.__name__] = []
        i = 0
        for hyperparams in tqdm(product(*model_hyperparameters.values())):
            kwargs = dict(zip(model_hyperparameters.keys(), hyperparams))
            if model_class.__name__ == "SVMClassifier":
                if kwargs["kernel"] != "poly" and kwargs["degree"] != 3:
                    continue
            i += 1
            if i < len(accuracies[model_class.__name__]):
                continue
            model = model_class(**kwargs)
            try:
                with time_limit(limit):
                    model.fit(X_train, y_train)
                    
                acc = np.mean(model.predict(X_test) == np.array(y_test))
                accuracies[model_class.__name__].append((acc, kwargs))
            except TimeoutException as e:
                acc = np.NaN
                
            with open(path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([[model_class.__name__, acc, kwargs]])
            if model_class.__name__ == 'TabRClassifier':
                del model
                gc.collect()
                torch.cuda.empty_cache()
                
    return accuracies


def cross_validate_models(models, kwargs_lists, X, y, path, limit):
    accuracies = {}
    kf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
    if not Path(path).is_file():
        with open(path, "w", newline="") as csvfile:
            fieldnames = ["model_class", "acc", "kwargs"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    else:
        df = pd.read_csv(path)
        for m, group in df.groupby('model_class'):
            accuracies[m] = list(zip(group['acc'].tolist(), group['kwargs'].apply(literal_eval).tolist()))
    for model_class in models:
        if model_class.__name__ not in accuracies:
            accuracies[model_class.__name__] = []
        for i, kwargs in tqdm(enumerate(kwargs_lists[model_class.__name__])):
            if i < len(accuracies[model_class.__name__]):
                continue
            model = model_class(**kwargs)
            try:
                s=0
                for train_idx, test_idx in kf.split(X, y):
                    X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
                    X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
                    with time_limit(limit):
                        model.fit(X_train, y_train)
                    s += np.mean(model.predict(X_test) == np.array(y_test))
                acc = s / NUM_SPLITS
                accuracies[model_class.__name__].append((acc, kwargs))
            except TimeoutException as e:
                acc = np.NaN
                
            with open(path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([[model_class.__name__, acc, kwargs]])
            if model_class.__name__ == 'TabRClassifier':
                del model
                gc.collect()
                torch.cuda.empty_cache()

                
    return accuracies


def get_top_kwargs(models, accuracies, cutoff):
    top_kwargs = {}
    for model_class in models:
        _, top_models_ = zip(*(sorted(accuracies[model_class.__name__], key=lambda x: x[0])[-int(cutoff*len(accuracies[model_class.__name__])):]))
        top_kwargs[model_class.__name__] = list(top_models_)
    return top_kwargs


def get_rashomon_sets(models, hyperparameters, X, y, initial_cutoff, top, initial_path=MODEL_ACCURACIES_PATH,
                      initial_accuracies_complete_path = INITIAL_ACCURACIES_PATH,
                      cross_validation_path=FILTERED_MODEL_ACCURACIES_PATH, initial_time_limit=TIME_LIMIT,
                      cross_validation_time_limit=TIME_LIMIT_CROSS_VALIDATION):
    assert top <= initial_cutoff
    initial_accuracies = train_test_models(models, hyperparameters, X, y, path=initial_path, limit=initial_time_limit)
    pickle.dump(initial_accuracies, open(initial_accuracies_complete_path, 'wb'))
    filtered_kwargs_lists = get_top_kwargs(models, initial_accuracies, initial_cutoff)
    final_accuracies = cross_validate_models(models, filtered_kwargs_lists, X, y, path=cross_validation_path,
                                             limit=cross_validation_time_limit)
    top_kwargs = get_top_kwargs(models, final_accuracies, top/initial_cutoff)
    return top_kwargs

