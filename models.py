import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler # To preprocess data for SVM - greatly improves performance
from sklearn.svm import SVC
from pytorch_tabr import TabRClassifier as TabRClassifier_


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def SVMClassifier(**hyperparams):
    return make_pipeline(StandardScaler(), SVC(**hyperparams))


class TabRClassifier(TabRClassifier_):
    def __init__(self, **kwargs):
        # selection_function_name="sparsemax",
        # context_dropout=0.5,
        # context_sample_size=2000,
        # num_embeddings={"type": "PLREmbeddings", "n_frequencies": 32, "frequency_scale": 32, "d_embedding": 32, "lite": False},
        super().__init__(**kwargs)
        self.type_embeddings="one-hot"
        self.device_name=DEVICE
        self.optimizer_params={"lr": 2e-4}
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        **kwargs
    ) -> None:
        super().fit(X_train=X_train.values, y_train=y_train.values,
                    max_epochs=1, batch_size=25, **kwargs)
        
    def predict(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            return super().predict(X=X.values)
        else:
            return super().predict(X=X)
    
    def predict_proba(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            return super().predict_proba(X=X.values)
        else:
            return super().predict_proba(X=X)