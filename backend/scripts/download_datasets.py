import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from joblib import Parallel, delayed
import os, random, string


# ============================================================
# 1. Realistic row distribution
# ============================================================
def generate_realistic_rows():
    r = random.random()
    if r < 0.40:
        return random.randint(1000, 20000)            # small
    elif r < 0.80:
        return random.randint(20000, 100000)          # medium
    else:
        return random.randint(100000, 500000)         # large


# ============================================================
# 2. Realistic feature names
# ============================================================
def generate_feature_names(n_features):
    base = [
        "temperature", "humidity", "pressure", "speed", "distance",
        "age", "income", "rating", "score", "duration", "altitude",
        "lat", "lon", "height", "weight", "volume"
    ]
    names = base.copy()
    while len(names) < n_features:
        names.append(random.choice(base) + "_" + str(random.randint(1, 9999)))
    return names[:n_features]


# ============================================================
# 3. Safe dataset generator (NEVER CRASHES)
# ============================================================
def generate_single_dataset(idx, save_dir):

    try:
        n_samples = generate_realistic_rows()
        n_features = random.randint(10, 40)
        n_classes = random.choice([2, 3, 4, 5, 10])

        # -----------------------------------------
        # *** THE IMPORTANT FIX ***
        # n_informative MUST be >= ceil(log2(n_classes))
        # -----------------------------------------
        min_informative = int(np.ceil(np.log2(n_classes)))
        n_informative = random.randint(min_informative, min(10, n_features - 4))

        # Ensure valid counts
        remaining = n_features - n_informative
        n_redundant = random.randint(0, max(0, remaining - 2))
        n_repeated  = random.randint(0, max(0, remaining - n_redundant))

        # Clusters per class must be valid
        max_clusters = 2 ** n_informative
        n_clusters = max(1, max_clusters // n_classes)

        # -----------------------------------------
        # SAFEST POSSIBLE MAKE_CLASSIFICATION CALL
        # -----------------------------------------
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters,
            flip_y=random.uniform(0.00, 0.10),
            class_sep=random.uniform(0.8, 2.5),
            random_state=random.randint(1, 999999),
        )

        # Convert to dataframe
        feature_names = generate_feature_names(n_features)
        df = pd.DataFrame(X, columns=feature_names)

        # Add categorical column
        df["category_col"] = np.random.choice(
            ["A", "B", "C", "D", "E"], size=n_samples,
            p=[0.4, 0.2, 0.2, 0.1, 0.1]
        )

        # Add random id column
        df["random_id"] = [
            ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            for _ in range(n_samples)
        ]

        df["target"] = y

        filepath = os.path.join(save_dir, f"{idx}_classification_dataset.csv")
        df.to_csv(filepath, index=False)

        print(f"✅ Saved: {filepath}  |  shape={df.shape}")

    except Exception as e:
        print(f"❌ Error generating dataset {idx}: {e}")


# ============================================================
# 4. Parallel execution
# ============================================================
def generate_parallel(n_datasets=100):
    save_dir = "datasets/synthetic_classification"
    os.makedirs(save_dir, exist_ok=True)

    Parallel(n_jobs=-1)(
        delayed(generate_single_dataset)(i, save_dir)
        for i in range(n_datasets)
    )

    print("\n🎉 Finished generating datasets with ZERO crashes.\n")


# ============================================================
# 5. Run
# ============================================================
if __name__ == "__main__":
    generate_parallel(n_datasets=100)
