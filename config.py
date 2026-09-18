import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Flask secret key (used to sign session cookies).
    # In a real deployment this should come from an environment variable.
    SECRET_KEY = os.environ.get("SECRET_KEY", "expenseai-dev-secret-key-change-me")

    # SQLite database lives inside the /database folder.
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(
        BASE_DIR, "database", "expense_tracker.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ML model file paths (used in later phases).
    ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "expense_model.pkl")
    ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "models", "anomaly_model.pkl")
    SPENDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "spending_model.pkl")

    # Categories used throughout the app (dropdowns, ML labels, charts).
    CATEGORIES = [
        "Food",
        "Transport",
        "Bills",
        "Entertainment",
        "Shopping",
        "Health",
        "Education",
        "Other",
    ]
