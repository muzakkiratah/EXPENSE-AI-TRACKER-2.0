"""
ExpenseAI - Phase 11
Train the anomaly (unusual spending) detection model.

Concept:
    This is UNSUPERVISED learning - there are no "correct answers" to learn
    from. Isolation Forest instead learns what "normal" spending generally
    looks like and flags expenses that don't fit that pattern as unusual.

Feature engineering (IMPORTANT - read before changing):
    An earlier version of this script fed the model [amount, day_of_month,
    one-hot category columns] directly. That failed badly: categories with
    few historical examples (Apparel: 7, Household: 6, Social Life: 5, out
    of 227 total rows) are themselves rare combinations in the full
    dataset, so Isolation Forest learned to isolate ANY point in those
    categories almost immediately - regardless of the amount. A real
    historical Social Life expense of Rs.150 got flagged as "unusual" purely
    for being in a rarely-seen category, not because Rs.150 is actually an
    unusual amount for Social Life.

    The fix: instead of raw amount + one-hot category, compute how many
    standard deviations each amount is from ITS OWN category's mean
    (a z-score), and feed that single normalized value to the model. This
    directly captures "is this amount unusual for this category" rather
    than "is this category rare in the dataset overall" - which is what we
    actually want to detect.

Input:
    ml/expense_data_1.csv   (the user's original dataset)

Output:
    models/anomaly_model.pkl
        A dict containing:
          - "model": the trained IsolationForest (1D: z-score, day_of_month)
          - "category_stats": {category: {"mean": ..., "std": ...}} used to
            compute the z-score consistently at prediction time
          - "global_mean" / "global_std": fallback for a category with no
            historical stats at all

Note on how this is actually used in the app:
    A single global model trained once on historical data is a reasonable
    "cold start" baseline, but real anomaly detection should ideally reflect
    each user's OWN spending habits (Rs.2,500 might be normal for one person
    and wildly unusual for another). app.py therefore uses THIS global model
    as one signal, combined with a lightweight per-user statistical check
    (comparing a new expense to that user's own historical mean/std in the
    same category) once they have enough transaction history. See
    `detect_anomaly()` in app.py.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RAW_DATA_PATH = os.path.join(BASE_DIR, "expense_data_1.csv")
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "anomaly_model.pkl")

TARGET_CATEGORIES = [
    "Food",
    "Transport",
    "Bills",
    "Entertainment",
    "Shopping",
    "Health",
    "Education",
    "Other",
]

# Maps the raw dataset's original category labels onto our new taxonomy -
# must stay identical to the mapping in create_training_dataset.py.
RAW_CATEGORY_MAP = {
    "Food": "Food",
    "Transportation": "Transport",
    "Household": "Bills",
    "Social Life": "Entertainment",
    "Apparel": "Shopping",
    "Education": "Education",
    "Other": "Other",
}

RANDOM_STATE = 42
CONTAMINATION = 0.07  # assume roughly 7% of historical expenses are unusual
MIN_STD_FLOOR_RATIO = 0.25  # a category's std is never allowed below 25% of its mean,
                             # so categories with very few/identical samples don't get
                             # a near-zero std that makes everything look "unusual"


def load_raw_expenses():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {RAW_DATA_PATH}. Place expense_data_1.csv in the ml/ folder."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    df = df[df["Income/Expense"] == "Expense"]
    df = df.copy()
    df["Category"] = df["Category"].map(RAW_CATEGORY_MAP)
    df = df.dropna(subset=["Category"])
    df = df[df["Category"].isin(TARGET_CATEGORIES)]
    df = df.dropna(subset=["INR", "Category", "Date"])

    normalized_dates = df["Date"].astype(str).str.replace("-", "/", regex=False)
    df = df.copy()
    df["parsed_date"] = pd.to_datetime(normalized_dates, errors="coerce")
    df = df.dropna(subset=["parsed_date"])

    df["day_of_month"] = df["parsed_date"].dt.day
    df["amount"] = df["INR"].astype(float)
    return df


def compute_category_stats(df):
    """
    Per-category mean/std, with a floor on std so categories with very few
    or near-identical historical amounts don't end up with a tiny std that
    would make every future amount look like a huge z-score outlier.

    The floor is WIDER for categories with very little historical data
    (e.g. Education here has only 1 real example) - a mean/std estimated
    from 1-3 points is inherently unreliable, so we deliberately give those
    categories more tolerance before flagging anything as unusual, rather
    than let a shaky single-sample estimate drive false positives.
    """
    stats = {}
    for cat in TARGET_CATEGORIES:
        amounts = df[df["Category"] == cat]["amount"]
        if len(amounts) == 0:
            continue
        n = len(amounts)
        mean = float(amounts.mean())
        std = float(amounts.std()) if n > 1 else 0.0

        if n <= 3:
            floor_ratio = 0.5
        elif n <= 10:
            floor_ratio = 0.35
        else:
            floor_ratio = MIN_STD_FLOOR_RATIO

        std = max(std, mean * floor_ratio, 1.0)
        stats[cat] = {"mean": mean, "std": std, "n": n}
    return stats


def train():
    print("=" * 60)
    print("ExpenseAI - Anomaly Detection Model Training")
    print("=" * 60)

    df = load_raw_expenses()
    print(f"\nUsable historical expense rows: {len(df)}")

    category_stats = compute_category_stats(df)
    print("\nPer-category stats (mean, std, sample count):")
    for cat, s in category_stats.items():
        print(f"  {cat:<15} mean=Rs.{s['mean']:.0f}  std=Rs.{s['std']:.0f}  n={s['n']}")

    global_mean = float(df["amount"].mean())
    global_std = max(float(df["amount"].std()), global_mean * MIN_STD_FLOOR_RATIO, 1.0)

    def zscore(row):
        stats = category_stats.get(row["Category"], {"mean": global_mean, "std": global_std})
        return (row["amount"] - stats["mean"]) / stats["std"]

    df = df.copy()
    df["amount_zscore"] = df.apply(zscore, axis=1)

    features = df[["amount_zscore", "day_of_month"]]
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"amount_zscore stats:\n{features['amount_zscore'].describe()}")

    model = IsolationForest(
        n_estimators=100,
        max_samples=min(256, len(features)),
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining IsolationForest (contamination={CONTAMINATION})...")
    model.fit(features)

    predictions = model.predict(features)   # -1 = anomaly, 1 = normal
    scores = model.decision_function(features)  # lower = more unusual

    df["anomaly_flag"] = predictions == -1
    df["anomaly_score"] = scores

    num_anomalies = int(df["anomaly_flag"].sum())
    print(f"\nFlagged {num_anomalies} out of {len(df)} historical expenses as unusual "
          f"({num_anomalies / len(df) * 100:.1f}%).")

    print("\nTop 10 most unusual historical expenses (lowest score = most unusual):")
    top_unusual = df.sort_values("anomaly_score").head(10)
    print(top_unusual[["Category", "amount", "amount_zscore", "day_of_month", "anomaly_score"]]
          .to_string(index=False))

    print("\nAverage amount: normal vs flagged-as-unusual expenses")
    print(df.groupby("anomaly_flag")["amount"].mean())

    payload = {
        "model": model,
        "category_stats": category_stats,
        "global_mean": global_mean,
        "global_std": global_std,
    }

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(payload, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
