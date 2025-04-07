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

    def add_interaction_features(self, X):
        X_interactions = X.copy()
        feature_pairs = list(combinations(original_features, 2))
        print(f"Adding {len(feature_pairs)} interaction features")
        
        for col1, col2 in feature_pairs:
            interaction_name = f"{col1}_{col2}"
            X_interactions[interaction_name] = X[col1] * X[col2]
        
        return X_interactions

    def add_shif_features(self, X, shifts=[1, 3], features=None):
        X_shifted = X.copy()
        X_shifted = X_shifted.sort_index()
        
        if features is None:
            features = [col for col in X.columns if col != "year"]
        
        total_new_features = 0
        for shift in shifts:
            for feature in features:
                shift_name = f"{feature}_lag_{shift}"
                X_shifted[shift_name] = X_shifted[feature].shift(shift)
                total_new_features += 1
        
        X_shifted = X_shifted.fillna(0)
        print(f"Added {total_new_features} shift features")
        return X_shifted

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

class LogisticRegressionTrainer:
    def __init__(self, X, y, model_name="LogisticRegression"):
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.X_unscaled = X.copy()
        self.X = pd.DataFrame(
            self.scaler.fit_transform(X.drop("year", axis=1)),
            columns=X.drop("year", axis=1).columns
        )
        self.X["year"] = X["year"].values
        self.y = y.reset_index(drop=True)
        self.best_model = None
        self.feature_names = X.drop("year", axis=1).columns

    def train(self):
        print(f"\n=== Training {self.model_name} ===")
        start_time = time.time()
        
        param_dist = {
            "C": uniform(0.001, 10),
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["saga"],
            "l1_ratio": uniform(0, 1)
        }

        n_splits = self.X_unscaled["year"].nunique()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        search = RandomizedSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_distributions=param_dist,
            n_iter=30,
            scoring="roc_auc",
            cv=tscv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        X_train = self.X.drop("year", axis=1)
        search.fit(X_train, self.y)
        self.best_model = search.best_estimator_
        
        elapsed_time = time.time() - start_time
        print(f"[{self.model_name}] Training completed in {elapsed_time:.2f} seconds")
        print(f"[{self.model_name}] Best Parameters: {search.best_params_}")
        print(f"[{self.model_name}] Best CV Score: {search.best_score_:.4f}")
        
        if hasattr(self.best_model, "coef_"):
            coefficients = self.best_model.coef_[0]
            feature_importance = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": np.abs(coefficients)
            }).sort_values("Importance", ascending=False)
            print(f"\n[{self.model_name}] Feature Importance:")
            print(feature_importance.head(10))
        
        return self.best_model

    def evaluate_cv(self, reporter):
        print(f"\n=== Evaluating {self.model_name} with Cross-Validation ===")
        n_splits = self.X_unscaled["year"].nunique()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        X_for_cv = self.X.drop("year", axis=1)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_unscaled)):
            X_val = X_for_cv.iloc[val_idx]
            y_val = self.y.iloc[val_idx]
            y_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
            y_pred_class = (y_pred_proba >= 0.5).astype(int)

            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)

            precision = precision_score(y_val, y_pred_class)
            recall = recall_score(y_val, y_pred_class)
            f1 = f1_score(y_val, y_pred_class)

            cm = confusion_matrix(y_val, y_pred_class)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            ax.set_title(f"[{self.model_name}] Fold {fold + 1}")

            reporter.log_confusion_matrix(self.model_name, fold + 1, fig, precision, recall, f1)
            plt.close(fig)

        print(f"[{self.model_name}] CV AUC scores: {[f"{s:.4f}" for s in scores]}")
        print(f"[{self.model_name}] Mean CV AUC: {np.mean(scores):.4f}")
        print(f"[{self.model_name}] Std CV AUC: {np.std(scores):.4f}")
        return scores


class GBDTTrainer:
    def __init__(self, X, y, model_name="XGBoost"):
        self.model_name = model_name
        self.X = X.copy()
        self.X_for_train = X.drop("year", axis=1).reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.best_model = None
        self.feature_names = X.drop("year", axis=1).columns

    def train(self):
        print(f"\n=== Training {self.model_name} ===")
        start_time = time.time()
        
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42
        )

        param_dist = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4)
        }

        n_splits = self.X["year"].nunique()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=30,
            scoring="roc_auc",
            cv=tscv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(self.X_for_train, self.y)
        self.best_model = search.best_estimator_
        
        elapsed_time = time.time() - start_time
        print(f"[{self.model_name}] Training completed in {elapsed_time:.2f} seconds")
        print(f"[{self.model_name}] Best Parameters: {search.best_params_}")
        print(f"[{self.model_name}] Best CV Score: {search.best_score_:.4f}")
        
        feature_importance = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": self.best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        print(f"\n[{self.model_name}] Feature Importance:")
        print(feature_importance.head(10)) 
        
        return self.best_model
    
    def evaluate_cv(self, reporter):
        print(f"\n=== Evaluating {self.model_name} with Cross-Validation ===")
        n_splits = self.X["year"].nunique()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            X_val = self.X_for_train.iloc[val_idx]
            y_val = self.y.iloc[val_idx]
            y_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
            y_pred_class = (y_pred_proba >= 0.5).astype(int)

            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)

            precision = precision_score(y_val, y_pred_class)
            recall = recall_score(y_val, y_pred_class)
            f1 = f1_score(y_val, y_pred_class)

            cm = confusion_matrix(y_val, y_pred_class)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            ax.set_title(f"[{self.model_name}] Fold {fold + 1}")

            reporter.log_confusion_matrix(self.model_name, fold + 1, fig, precision, recall, f1)
            plt.close(fig)

        print(f"[{self.model_name}] CV AUC scores: {[f"{s:.4f}" for s in scores]}")
        print(f"[{self.model_name}] Mean CV AUC: {np.mean(scores):.4f}")
        print(f"[{self.model_name}] Std CV AUC: {np.std(scores):.4f}")
        return scores            

