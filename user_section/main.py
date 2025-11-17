import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from user_section.training.user_regression_training import UserRegressionTrainer
from user_section.training.user_classification_training import UserClassificationTrainer
from user_section.prediction.regression_prediction import MetaRegressionPredictor
from user_section.prediction.classification_prediction import (
    MetaClassificationPredictor,
)
from constants import *


class User:
    def __init__(
        self, dataset_path, user_id, target_col, tuning, task_type, dataset_name
    ):
        self.dataset_path = dataset_path
        self.user_id = user_id
        self.target_col = target_col
        self.tuning = tuning
        self.task_type = task_type
        self.dataset_name = dataset_name

    def regression(self):
        predictor = MetaRegressionPredictor(
            dataset_path=self.dataset_path,
            tuning=self.tuning,
            target_col=self.target_col,
        )

        predictor.run_pipeline()

        trainer = UserRegressionTrainer(
            predictor, self.user_id, self.tuning, self.dataset_name
        )

        output = trainer.train_and_tune_model()
        print(output)

    def classification(self):
        predictor = MetaClassificationPredictor(
            dataset_path=self.dataset_path,
            tuning=self.tuning,
            target_col=self.target_col,
        )

        predictor.run_pipeline()

        trainer = UserClassificationTrainer(predictor, self.user_id, self.dataset_name)

        output = trainer.train_and_tune_model(self.tuning)

        print(output)

    def start(self):
        if self.task_type == "regression":
            try:
                self.regression()
                new_entry = {
                    "user_id": self.user_id,
                    "dataset_name": self.dataset_name,
                    "dataset_path": self.dataset_path,
                    "target_col": self.target_col,
                }

                os.makedirs(
                    os.path.dirname(PENDING_DATSETS_REGRESSION_FILE), exist_ok=True
                )

                df = pd.read_csv(PENDING_DATSETS_REGRESSION_FILE)
                df = pd.concat([df, pd.DataFrame([new_entry])], axis=0, ignore_index=True)
                df.to_csv(PENDING_DATSETS_REGRESSION_FILE,index=False)
            except Exception as e:
                print(
                    "We are not able to process your dataset right now!! Our best minds are working on it "
                )
                print(e)

        elif self.task_type == "classification":
            try:

                self.classification()
                new_entry = {
                    "user_id": self.user_id,
                    "dataset_name": self.dataset_name,
                    "dataset_path": self.dataset_path,
                    "target_col": self.target_col,
                }

                os.makedirs(
                    os.path.dirname(PENDING_DATSETS_CLASSIFICATION_FILE), exist_ok=True
                )

                df = pd.read_csv(PENDING_DATSETS_CLASSIFICATION_FILE)
                df = pd.concat([df, pd.DataFrame([new_entry])], axis=0, ignore_index=True)
                df.to_csv(PENDING_DATSETS_CLASSIFICATION_FILE,index=False)
            except Exception as e:
                print(
                    "We are not able to process your dataset right now!! Our best minds are working on it "
                )
                print(e)


        else:
            return "Error with task!"


if __name__ == "__main__":
    user = User(
        "datasets/classification/synthetic.csv",
        user_id=1,
        target_col="target",
        tuning=False,
        task_type="classification",
        dataset_name="synthetic",
    )
    
    user.start()
