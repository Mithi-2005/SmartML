import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

from constants import *
import pandas as pd
import numpy as np
from joblib import dump
from components.preprocessing import Preproccessor
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Lasso,
    Ridge,
    ElasticNet,
)
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from user_section.prediction.classification_prediction import MetaClassificationPredictor


class UserClassificationTrainer:

    def __init__(
        self,
        predictor:MetaClassificationPredictor,
        user_id,
        dataset_name,
        task_type
    ):

        self.best_models = predictor.top_models
        self.X_train = predictor.X_train
        self.y_train = predictor.y_train
        self.X_test = predictor.X_test
        self.y_test = predictor.y_test
        self.X_val = predictor.X_val
        self.y_val = predictor.y_val
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.task_type = task_type

        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1),
            "KNN": KNeighborsClassifier(n_jobs=-1),
            "DecisionTree": DecisionTreeClassifier(),
            "SVC": SVC(),
            "RandomForest": RandomForestClassifier(n_jobs=-1),
            "GradientBoosting": HistGradientBoostingClassifier(early_stopping=False),
        }

        self.param_grids = {
            "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
            "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
            "DecisionTree": {"max_depth": [5, 10, None]},
            "SVC": {"C": [0.1, 1, 10]},
            "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
            "GradientBoosting": {
                "learning_rate": [0.05, 0.1],
                "max_iter": [200, 300],
                "max_leaf_nodes": [31, 63],
            },
        }

    
    def train_and_tune_model(self, Tuning=False):

        if(self.task_type!="classification"):
            raise ValueError("Task type is not classification !")
            

        all_results = []

        for model in self.best_models:

            print(f"[TRAIN] {model} ")

            base_model = self.models[model]

            params = self.param_grids[model]

            if Tuning:

                grid = GridSearchCV(
                    estimator=base_model,
                    param_grid=params,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                )

                grid.fit(self.X_train, self.y_train)

                best_est = grid.best_estimator_
                val_acc = best_est.score(self.X_val, self.y_val)

                all_results.append({"Estimator": best_est, "Accuracy": val_acc})
            else:
                base_model.fit(self.X_train, self.y_train)
                new_test_x = pd.concat([self.X_val, self.X_test], axis=0)
                new_test_y = pd.concat([self.y_val, self.y_test], axis=0)
                val_acc = base_model.score(new_test_x, new_test_y)
                all_results.append({"Estimator": base_model, "Accuracy": val_acc})

        final_model = pd.DataFrame(all_results).sort_values("Accuracy", ascending=False)
        best_row = final_model.iloc[0]

        print(
            f"\n[SELECT] Best Model: {best_row['Estimator']} (Accuracy={best_row['Accuracy']:.4f})"
        )

        path = f"{USERS_FOLDER}/{self.user_id}/models/classification"
        os.makedirs(path, exist_ok=True)
        save_path = f"{path}/{self.dataset_name}.pkl"
        dump(best_row["Estimator"], save_path)
        print(f"💾 Saved Best Model → {save_path}")

        return {
            "best_model_name": best_row['Estimator'],
            "best_model_path": save_path,
            "all_results": all_results,
        }



