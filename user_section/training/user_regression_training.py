import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))
from joblib import dump
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, Any, Optional, List
from constants import *
from user_section.prediction.regression_prediction import MetaRegressionPredictor
class UserRegressionTrainer:
    def __init__(
        self,
        predictor:MetaRegressionPredictor,
        user_id,
        tuning: bool,
        dataset_name
    ):
        self.models_list = {
            "LinearRegression": LinearRegression(),
            "PolynomialRegression": Pipeline(
                [
                    ("PolyFeatures", PolynomialFeatures(degree=2)),
                    ("regressor", LinearRegression()),
                ]
            ),
            "Ridge": Ridge(),
            "Lasso": Lasso(max_iter=10000),
            "ElasticNet": ElasticNet(max_iter=10000),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "SVR": SVR(),
            "LinearSVR": LinearSVR(),
            "RandomForest": RandomForestRegressor(n_jobs=-1),
            "GradientBoosting": HistGradientBoostingRegressor(early_stopping=False),
        }

        self.params = {
            "LinearRegression": {},
            "PolynomialRegression": {
                "PolyFeatures__degree": [2, 3],
                "regressor__fit_intercept": [True, False],
            },
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
                "solver": ["auto", "lbfgs", "saga"],
            },
            "Lasso": {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "max_iter": [5000, 10000],
            },
            "ElasticNet": {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_iter": [5000, 10000],
            },
            "KNN": {
                "n_neighbors": [3, 5, 7, 9, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            "DecisionTree": {
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [None, "sqrt", "log2"],
            },
            "SVR": {
                "kernel": ["rbf", "poly", "linear"],
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto"],
                "degree": [2, 3, 4],
            },
            "LinearSVR": {
                "C": [0.1, 1.0, 10.0],
                "epsilon": [0.0, 0.1, 0.3],
                "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
                "max_iter": [2000, 5000, 10000],
            },
            "RandomForest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False],
            },
            "GradientBoosting": {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [None, 6, 12, 20],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [20, 40, 60],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        }

        self.best_models = predictor.top_models
        self.X_train = predictor.X_train
        self.X_test = predictor.X_test
        self.X_val = predictor.X_val
        self.y_train = predictor.y_train
        self.y_test = predictor.y_test
        self.y_val = predictor.y_val
        self.tuning = tuning
        self.user_id = user_id
        self.dataset_name=dataset_name

    def train_and_tune_model(self):

        results = {}

        print(f"\n==============================")
        print(f"👤 User: {self.user_id}")
        print(f"🏆 Training Top-{len(self.best_models)} Models")
        print(f"==============================\n")

        if not self.tuning:
            print("ℹ No tuning → Using full training data (test + val)")
            self.X_test = pd.concat([self.X_test, self.X_val], axis=0)
            self.y_test = pd.concat([self.y_test, self.y_val], axis=0)

        for model_name in self.best_models:

            print(f"\n🚀 Training Model: {model_name}")
            model = self.models_list[model_name]

            if (
                self.tuning
                and model_name in self.params
                and len(self.params[model_name]) > 0
            ):
                print(f"🔧 Tuning {model_name}...")
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=self.params[model_name],
                    n_iter=20,
                    cv=5,
                    scoring="r2",
                    n_jobs=-1,
                    random_state=42,
                    verbose=1,
                )
                search.fit(self.X_train, self.y_train)
                best_model = search.best_estimator_
                print(f"✅ Best Params: {search.best_params_}")
            else:

                print(f"⚡ Training {model_name} on full data (train + val)")
                model.fit(self.X_train, self.y_train)
                best_model = model

          
            y_train_pred = best_model.predict(self.X_train)
            y_test_pred = best_model.predict(self.X_test)

            # val only for tuning mode
            val_r2 = None
            if self.tuning:
                y_val_pred = best_model.predict(self.X_val)
                val_r2 = r2_score(self.y_val, y_val_pred)

            metrics = {
                "train_r2": r2_score(self.y_train, y_train_pred),
                "val_r2": val_r2,  
                "test_r2": r2_score(self.y_test, y_test_pred),
            }

            print(f"📊 {model_name} Results: {metrics}")

            results[model_name] = {"model": best_model, "metrics": metrics}

    
        if self.tuning:
            # Use val_r2 for tuning scenario
            best_model_name = max(
                results, key=lambda m: results[m]["metrics"]["val_r2"]
            )
        else:
            # In no-tuning scenario → use test_r2 for selection
            best_model_name = max(
                results, key=lambda m: results[m]["metrics"]["test_r2"]
            )

        best_model = results[best_model_name]["model"]

        print(f"\n🎖 Best Model: {best_model_name}")

        
        path=f"{USERS_FOLDER}/{self.user_id}/models/regression"
        os.makedirs(path, exist_ok=True)
        save_path = f"{path}/{self.dataset_name}.pkl"
        dump(best_model, save_path)
        print(f"💾 Saved Best Model → {save_path}")

        return {
            "best_model_name": best_model_name,
            "best_model_path": save_path,
            "all_results": results,
        }
