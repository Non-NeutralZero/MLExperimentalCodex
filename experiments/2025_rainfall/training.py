import warnings
warnings.filterwarnings("ignore")

import os
import time
import multiprocessing
from contextlib import contextmanager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

from scipy.stats import uniform, randint
import xgboost as xgb

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    pool.join()
    pool.close()


# Original features
original_features = ["day", "pressure", "maxtemp", "temparature", "mintemp", "dewpoint", 
                     "humidity", "cloud", "sunshine", "winddirection", "windspeed"]

class DataHandler:
    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self._add_years()

    def _add_years(self):
        self.train["year"] = self.train["id"] // 365
        self.test["year"] = self.test["id"] // 365
        print(f"The train dataset contains {self.train.year.nunique()} years")

    def get_features_and_target(self):
        data = self.train.sort_values("id").reset_index(drop=True)
        X = data.drop(["id", "rainfall"], axis=1).fillna(0)
        y = data["rainfall"]
        return X, y

    def get_test_features(self):
        data = self.test.sort_values("id").reset_index(drop=True)
        X = data.drop(["id"], axis=1).fillna(0)
        return X, self.test["id"]

class RFECVSelector:
    def __init__(self, X, y, model_type="lr"):
        self.X = X.copy()
        self.y = y.copy()
        self.cv = X["year"].nunique()
        self.model_type = model_type
        self.selected_features = None
        self.rfecv = None
        
    def select_features(self):
        print(f"\n=== Selecting features using RFECV with {self.model_type.upper()} ===")
        
        X_for_selection = self.X.drop("year", axis=1)
        feature_names = X_for_selection.columns
        
        n_splits = min(self.X["year"].nunique(), self.cv)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # base model for feature selection
        if self.model_type == "lr":
            base_model = LogisticRegression(max_iter=1000, random_state=42, solver="saga", penalty="l2")
        elif self.model_type == "xgb":
            base_model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.rfecv = RFECV(
            estimator=base_model,
            step=1,
            cv=tscv,
            scoring="roc_auc",
            min_features_to_select=2,
            n_jobs=-1,  # multiprocessing.cpu_count()
            verbose=1
        )
        
        self.rfecv.fit(X_for_selection, self.y)
    
        selected_mask = self.rfecv.support_
        self.selected_features = feature_names[selected_mask].tolist()
        
        print(f"Optimal number of features: {self.rfecv.n_features_}")
        print(f"Selected features: {self.selected_features}")
        
        plt.figure()
        plt.title(f"RFECV - Optimal Number of Features ({self.model_type.upper()})")
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross-validation score (ROC AUC)")
        results_df = pd.DataFrame(self.rfecv.cv_results_)
        plt.errorbar(
            x=results_df["n_features"],
            y=results_df["mean_test_score"],
            yerr=results_df["std_test_score"],
        )
        plt.title("Recursive Feature Elimination with Cross-Validation")
        plt.tight_layout()
        plt.savefig(f"rfecv_{self.model_type}_feature_selection.png")
        plt.close()
        
        return self.selected_features
            
    def get_selected_feature_data(self, X=None):
        if self.selected_features is None:
            raise ValueError("You must run select_features() first")
        
        X_data = X if X is not None else self.X
        X_selected = X_data.copy()
        
        non_year_cols = [col for col in X_selected.columns if col != "year"]
        
        selected_non_year_cols = [col for col in non_year_cols if col in self.selected_features]
        
        # add year back for consistency with other processings
        selected_cols = selected_non_year_cols + ["year"]
        X_selected = X_selected[selected_cols]
        
        return X_selected

