import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler # To preprocess data for SVM - greatly improves performance
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pytorch_tabr import TabRClassifier as TabRClassifier_
from constants import DEVICE, SVMCLASSIFIER_HYPERPARAMETERS, TABRCLASSIFIER_HYPERPARAMETERS, XGBCLASSIFIER_HYPERPARAMETERS



def SVMClassifier(**hyperparams):
    return make_pipeline(StandardScaler(), SVC(**hyperparams))


class TabRClassifier(TabRClassifier_):
    def __init__(self, **kwargs):
        # selection_function_name="sparsemax",
        # context_dropout=0.5,
        # context_sample_size=2000,
        # num_embeddings={"type": "PLREmbeddings", "n_frequencies": 32, "frequency_scale": 32, "d_embedding": 32, "lite": False},
        self.max_epochs = kwargs.pop('max_epochs')
        super().__init__(**kwargs, device_name=DEVICE, verbose=1)
        self.type_embeddings="one-hot"
        self.optimizer_params={"lr": 2e-4}
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        **kwargs
    ) -> None:
        super().fit(X_train=X_train.values, y_train=y_train.values,
                    max_epochs=self.max_epochs, batch_size=25, **kwargs)
        
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
        

MODELS = [SVMClassifier, TabRClassifier, XGBClassifier]
HYPERPARAMETERS = {
    XGBClassifier.__name__ : XGBCLASSIFIER_HYPERPARAMETERS,
    SVMClassifier.__name__ : SVMCLASSIFIER_HYPERPARAMETERS,
    TabRClassifier.__name__ : TABRCLASSIFIER_HYPERPARAMETERS
}
