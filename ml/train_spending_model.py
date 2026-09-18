"""
ExpenseAI - Phase 12
Train the spending forecasting model.

Concept:
    Estimate a user's likely total spending for the current month, based on
    how their spending has trended over previous months (and, in the live
    app, how much they've already spent so far this month).

IMPORTANT - why this script's saved model isn't used directly at runtime:
    Every user has a different budget, different spending scale, and a
    different number of months of history. A single model trained once on
    one historical dataset (this script) can't meaningfully predict another
    user's numbers - a linear trend fitted to someone else's ₹3,000-20,000
    monthly range tells you nothing about a user who spends ₹500/month.

    So this script exists to:
      1. Demonstrate the regression technique end-to-end on real historical
         data (for your report/demo - dataset shape, train/test evaluation,
         MAE, sample predictions).
      2. Produce models/spending_model.pkl as a reference artifact.

    The LIVE forecasting feature in app.py (`forecast_monthly_spending()`)
    instead fits a fresh, lightweight LinearRegression on each user's own
    monthly totals at request time - this is the only version of "spending
    prediction" that's actually meaningful per-user, and it's what powers
    the dashboard forecast. It only produces a prediction once a user has
    enough of their own historical data; otherwise the dashboard shows
    "Not enough historical data for a reliable prediction," exactly as
    specified.

Input:
    ml/expense_data_1.csv

Output:
    models/spending_model.pkl
        A dict containing the trained LinearRegression model plus the
        month index -> total data it was trained on (for reference).
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RAW_DATA_PATH = os.path.join(BASE_DIR, "expense_data_1.csv")
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "spending_model.pkl")

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


def load_monthly_totals():
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
    df = df.dropna(subset=["INR", "Date"])

    normalized_dates = df["Date"].astype(str).str.replace("-", "/", regex=False)
    df = df.copy()
    df["parsed_date"] = pd.to_datetime(normalized_dates, errors="coerce")
    df = df.dropna(subset=["parsed_date"])

    df["month"] = df["parsed_date"].dt.to_period("M")
    monthly = df.groupby("month")["INR"].sum().sort_index()

    # The dataset's collection window doesn't start or end on a month
    # boundary (e.g. it starts mid-November and ends in early March), so the
    # first and last months in the grouped totals are PARTIAL, not full
    # calendar months. Feeding a partial month into the trend would badly
    # mislead the model (a half-month total looks like a huge spending drop).
    # Detect this using the actual min/max dates and drop any boundary month
    # that doesn't cover most of its calendar days.
    min_date, max_date = df["parsed_date"].min(), df["parsed_date"].max()
    months_index = monthly.index

    if len(months_index) > 0:
        first_month = months_index[0]
        if min_date.day > 5:  # dataset starts more than 5 days into the month
            print(f"  Dropping {first_month} - partial month (data starts on day {min_date.day}).")
            monthly = monthly.drop(first_month)

    if len(monthly.index) > 0:
        last_month = monthly.index[-1]
        days_in_last_month = last_month.days_in_month
        if max_date.day < days_in_last_month - 5:  # ends well before month-end
            print(f"  Dropping {last_month} - partial month (data ends on day {max_date.day}).")
            monthly = monthly.drop(last_month)

    return monthly


def train():
    print("=" * 60)
    print("ExpenseAI - Spending Forecast Model Training")
    print("=" * 60)

    monthly = load_monthly_totals()
    print(f"\nMonthly totals found ({len(monthly)} months):")
    print(monthly)

    if len(monthly) < 3:
        print(
            "\nNot enough historical months in expense_data_1.csv to train "
            "a meaningful demo regression (need at least 3). Skipping "
            "training - this is expected behavior, not an error: the spec "
            "explicitly says not to invent a prediction without enough data."
        )
        return

    values = monthly.values.astype(float)
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

    # Simple leave-last-out evaluation: train on all months except the last,
    # test the prediction against the actual last month.
    X_train, y_train = X[:-1], y[:-1]
    X_test, y_test = X[-1:], y[-1:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nLeave-last-month-out evaluation:")
    print(f"  Actual last month:    Rs. {y_test[0]:.2f}")
    print(f"  Predicted last month: Rs. {y_pred[0]:.2f}")
    print(f"  Absolute error:       Rs. {mae:.2f}")

    # Fit a final model on ALL available months for the saved artifact.
    final_model = LinearRegression()
    final_model.fit(X, y)
    r2 = r2_score(y, final_model.predict(X))
    print(f"\nFinal model (trained on all {len(values)} months) R^2 on training data: {r2:.3f}")
    print(
        "Note: R^2 on training data isn't a real generalization metric with "
        "this few points - it's reported for completeness, not as proof of "
        "accuracy. The leave-last-out MAE above is the more honest number."
    )

    next_index = np.array([[len(values)]])
    next_month_prediction = final_model.predict(next_index)[0]
    print(f"\nProjected next month's spending (demo): Rs. {next_month_prediction:.2f}")

    payload = {
        "model": final_model,
        "trained_on_months": [str(m) for m in monthly.index],
        "trained_on_values": values.tolist(),
    }

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(payload, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
