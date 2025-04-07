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

