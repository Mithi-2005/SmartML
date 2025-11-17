import sys, os
sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
from joblib import load
from components.meta_features_extraction import meta_features_extract_class
from components.preprocessing import Preproccessor
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from constants import *



class MetaModelClassification:
    def __init__(self, dataset_path, target_col):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.meta_model_path = META_CLASSIFICATION_MODEL

        self.user_dataset = pd.read_csv(dataset_path)
        self.preprocessor = Preproccessor(dataframe=dataset_path, target_col=target_col)

        self.meta_model = None
        self.meta_row = None
        self.explainer = None

    def preprocess(self):
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, self.task_type = self.preprocessor.run_preprocessing()
        return self

    def extract_meta_features(self):
        self.meta_row = meta_features_extract_class(
            X_train=self.X_train,
            y_train=self.y_train,
            best_model=None,
            raw_df=self.user_dataset,
            save=False
        )
        # keep as DataFrame with a single row
        if isinstance(self.meta_row, pd.Series):
            self.meta_row = self.meta_row.to_frame().T

        self.meta_row = self.meta_row.drop(columns=["best_model", "task_type"], errors="ignore")

    def load_meta_model(self):
        self.meta_model = load(self.meta_model_path)

    def get_probabilities(self):
        probs = self.meta_model.predict_proba(self.meta_row)[0]
        classes = self.meta_model.classes_
        out = [(c, p) for c, p in zip(classes, probs)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def predict_top2(self):
        probs = self.get_probabilities()
        return probs[:2]

    # # ---------- NEW: LIME EXPLANATIONS ----------

    # def build_lime_explainer(self):
    #     """
    #     Build a LIME Tabular explainer on the meta-feature space.
    #     For better results, you can pass the full meta-training dataset here
    #     instead of self.meta_row.
    #     """
    #     if self.meta_model is None:
    #         raise RuntimeError("Call load_meta_model() before building the explainer.")
    #     if self.meta_row is None:
    #         raise RuntimeError("Call extract_meta_features() before building the explainer.")

    #     self.explainer = LimeTabularExplainer(
    #         training_data=self.meta_row.values,  # ideally: meta-training features
    #         feature_names=self.meta_row.columns.tolist(),
    #         class_names=[str(c) for c in self.meta_model.classes_],
    #         mode="classification"
    #     )

    # def explain_top2_with_lime(self, num_features=10):
    #     """
    #     Returns a dict: { model_name: [(feature, weight), ...], ... }
    #     for the top-2 predicted models.
    #     """
    #     if self.explainer is None:
    #         raise RuntimeError("Call build_lime_explainer() before explaining.")

    #     # 1) Get top-2 model labels
    #     probs_sorted = self.get_probabilities()
    #     top2_models = [m for (m, _) in probs_sorted[:2]]

    #     # 2) Prepare instance (single meta-feature row)
    #     instance = self.meta_row.iloc[0].values

    #     # 3) Get LIME explanation for this instance
    #     exp = self.explainer.explain_instance(
    #         data_row=instance,
    #         predict_fn=self.meta_model.predict_proba,
    #         num_features=num_features,
    #         top_labels=2
    #     )

    #     # 4) Map from class label -> explanation list
    #     explanations = {}
    #     classes_list = list(self.meta_model.classes_)

    #     for model_name in top2_models:
    #         label_idx = classes_list.index(model_name)
    #         # as_list() gives [(feature_description, contribution), ...]
    #         explanations[model_name] = exp.as_list(label=label_idx)

    #     return explanations
    

if __name__ == "__main__":

    pipeline = MetaModelClassification(
        dataset_path="PreTraining/datasets/classification/gender_classification.csv",
        target_col="gender",
    )

    pipeline.preprocess()
    pipeline.extract_meta_features()
    pipeline.load_meta_model()

    # Top-2 recommended models
    best_models = pipeline.predict_top2()
    print("Top-2 recommended models and their probabilities:")
    for model_name, prob in best_models:
        print(f"{model_name}: {prob:.4f}")

    # # Build LIME explainer and get explanations
    # pipeline.build_lime_explainer()
    # explanations = pipeline.explain_top2_with_lime(num_features=10)

    # print("\nLIME explanations (feature contributions) for each of the top-2 models:\n")
    # for model_name, feats in explanations.items():
    #     print(f"=== Explanation for model: {model_name} ===")
    #     for feat, weight in feats:
    #         print(f"{feat:40s} -> {weight:+.4f}")
    #     print()

