import os
import traceback
import pandas as pd
from components.preprocessing import Preproccessor
from components.training import Classification_Training, Regression_Training

DATASET_FOLDER = "datasets/synthetic_classification"
LOG_FILE = "regression_run_log.txt"


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)  # prints immediately


def find_target_column(df, fallback="target"):
    """Try to guess target column automatically."""
    if fallback in df.columns:
        return fallback

    common = [
        # classification
        "class", "Class", "label", "Label", "target", "Target", "y", "Y",
        "binaryClass", "match", "CATEGORY",
        # regression common
        "value", "Value", "price", "Price", "amount", "Amount",
        "score", "Score", "output", "Output", "SalePrice"
    ]

    for c in common:
        if c in df.columns:
            return c

    return df.columns[-1]  # fallback


def main():
    datasets = sorted(
        [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
    )

    log(f"Found {len(datasets)} datasets.")
    log("----------------------------------------")

    for file in datasets:
        dataset_path = os.path.join(DATASET_FOLDER, file)

        log("\n\n========================")
        log(f"Running dataset: {file}")
        log("========================")

        try:
            df = pd.read_csv(dataset_path)

            # Skip huge datasets
            MAX_ROWS = 500000   # or whatever limit you choose
            if df.shape[0] > MAX_ROWS:
                log(f"⚠ Skipping {file} — too large ({df.shape[0]} rows).")
                continue

            target_col = find_target_column(df)

            log(f"Detected target_col = {target_col}")

            # Preprocessing
            preprocessor = Preproccessor(dataset_path, target_col)
            (
                X_train,
                y_train,
                X_test,
                y_test,
                X_val,
                y_val,
                task_type,
            ) = preprocessor.run_preprocessing()

            # Auto routing
            if task_type == "classification":
                log("📌 Task detected: CLASSIFICATION")
                trainer = Classification_Training(
                    X_train, y_train, X_test, y_test, X_val, y_val,
                    dataset_path, target_col
                )
                trainer.train_model()

            elif task_type == "regression":
                log("📌 Task detected: REGRESSION")
                trainer = Regression_Training(
                    X_train, y_train, X_test, y_test, X_val, y_val,
                    dataset_path, target_col
                )
                trainer.train_model()

            else:
                log("❌ Unknown task_type → skipping.")
                continue

            log(f"✔ SUCCESS: {file}")


        except Exception as e:
            log(f"❌ ERROR in {file}: {e}")
            log(traceback.format_exc())
            continue


if __name__ == "__main__":
    main()
