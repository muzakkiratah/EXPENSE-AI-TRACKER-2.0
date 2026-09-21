"""
ExpenseAI - ML-Based Smart Personal Expense Tracker
Phase 1-5: Project setup, database, authentication, dashboard, expense CRUD.

ML features (automatic categorization, anomaly detection, spending
prediction) are added in later phases - see the ml/ folder and the
project README for the plan.
"""

import os
import calendar
import numpy as np
from datetime import datetime, date
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, Response
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import extract, func

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)

# ---------------------------------------------------------------------------
# Try to load the ML category model. If it doesn't exist yet (Phase 6 hasn't
# been run), the app should still work - we just fall back to "Other".
# ---------------------------------------------------------------------------
category_model = None
try:
    import joblib
    if os.path.exists(app.config["ML_MODEL_PATH"]):
        category_model = joblib.load(app.config["ML_MODEL_PATH"])
        print("[ExpenseAI] Category ML model loaded successfully.")
    else:
        print("[ExpenseAI] No category model found yet at "
              f"{app.config['ML_MODEL_PATH']}. Run ml/train_model.py "
              "(Phase 6) to enable automatic categorization.")
except Exception as e:
    print(f"[ExpenseAI] Could not load category model: {e}")

# ---------------------------------------------------------------------------
# Try to load the ML anomaly detection model (Phase 11). If it doesn't exist
# yet, anomaly detection falls back to a simple per-user statistical check.
# ---------------------------------------------------------------------------
anomaly_model_payload = None
try:
    import joblib as _joblib  # noqa: F401 (already imported above if present)
    if os.path.exists(app.config["ANOMALY_MODEL_PATH"]):
        anomaly_model_payload = joblib.load(app.config["ANOMALY_MODEL_PATH"])
        print("[ExpenseAI] Anomaly detection model loaded successfully.")
    else:
        print("[ExpenseAI] No anomaly model found yet at "
              f"{app.config['ANOMALY_MODEL_PATH']}. Run "
              "ml/train_anomaly_model.py (Phase 11) to enable the global "
              "anomaly baseline. Per-user detection still works without it.")
except Exception as e:
    print(f"[ExpenseAI] Could not load anomaly model: {e}")


# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    monthly_budget = db.Column(db.Float, default=0.0)
    budget_threshold = db.Column(db.Integer, default=80)  # percent
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    expenses = db.relationship(
        "Expense", backref="user", lazy=True, cascade="all, delete-orphan"
    )


class Expense(db.Model):
    __tablename__ = "expenses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Phase 11: anomaly detection results, computed when the expense is
    # added or edited. Never set directly by the user.
    is_anomaly = db.Column(db.Boolean, default=False)
    anomaly_reason = db.Column(db.String(255), nullable=True)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        # The session can outlive the actual user row - e.g. if the
        # database file was deleted/recreated (as happens when picking up
        # a schema change) while a browser still has an old login cookie.
        # Without this check, routes further down would crash with
        # AttributeError: 'NoneType' object has no attribute 'id' the
        # moment they touched the (nonexistent) user.
        if current_user() is None:
            session.clear()
            flash("Your session has expired. Please log in again.", "error")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return User.query.get(user_id)


# ---------------------------------------------------------------------------
# Helper: predict category from description (falls back safely)
# ---------------------------------------------------------------------------
def predict_category(description: str) -> str:
    if not description or not description.strip():
        return "Other"
    if category_model is None:
        return "Other"
    try:
        prediction = category_model.predict([description])[0]
        return prediction
    except Exception as e:
        print(f"[ExpenseAI] Prediction failed: {e}")
        return "Other"


# ---------------------------------------------------------------------------
# Phase 11 - Anomaly (unusual spending) detection
#
# Two signals are combined:
#   1. GLOBAL model (Isolation Forest trained on historical data) - acts as
#      a baseline, useful especially for brand-new users with little history
#      of their own yet ("cold start").
#   2. PERSONAL statistical check - once a user has enough of their own
#      transactions in a category, compare the new expense against THEIR
#      own historical mean/std in that category. This is what actually
#      matches the spec's example ("₹2,500 is unusually high compared with
#      your previous Food spending of ₹150-250").
#
# If either signal flags the expense, it's marked as unusual. The personal
# check is only trusted once at least MIN_HISTORY_FOR_PERSONAL_CHECK past
# expenses exist in that category, so a single early "normal" expense isn't
# incorrectly flagged just for lack of comparison data.
# ---------------------------------------------------------------------------
MIN_HISTORY_FOR_PERSONAL_CHECK = 4
PERSONAL_ANOMALY_STD_MULTIPLIER = 2.0


def _global_anomaly_check(category: str, amount: float, expense_date: date) -> bool:
    if anomaly_model_payload is None:
        return False
    try:
        model = anomaly_model_payload["model"]
        category_stats = anomaly_model_payload["category_stats"]
        global_mean = anomaly_model_payload["global_mean"]
        global_std = anomaly_model_payload["global_std"]

        stats = category_stats.get(category, {"mean": global_mean, "std": global_std})
        zscore = (amount - stats["mean"]) / stats["std"]

        import pandas as pd
        feature_row = pd.DataFrame([{
            "amount_zscore": zscore,
            "day_of_month": expense_date.day,
        }])

        prediction = model.predict(feature_row)[0]  # -1 = anomaly, 1 = normal
        return prediction == -1
    except Exception as e:
        print(f"[ExpenseAI] Global anomaly check failed: {e}")
        return False


def _personal_anomaly_check(user, category: str, amount: float, exclude_expense_id=None):
    """
    Returns (is_anomaly: bool, reason: str or None).
    Only makes a call once enough personal history exists; otherwise
    returns (False, None) without judging.
    """
    query = Expense.query.filter_by(user_id=user.id, category=category)
    if exclude_expense_id:
        query = query.filter(Expense.id != exclude_expense_id)
    history = [e.amount for e in query.all()]

    if len(history) < MIN_HISTORY_FOR_PERSONAL_CHECK:
        return False, None

    mean = float(np.mean(history))
    std = float(np.std(history))

    if std == 0:
        threshold = mean * 2  # fallback if all past amounts were identical
    else:
        threshold = mean + (PERSONAL_ANOMALY_STD_MULTIPLIER * std)

    if amount > threshold:
        reason = (
            f"₹{amount:.0f} is unusually high for {category} compared with "
            f"your average of ₹{mean:.0f}."
        )
        return True, reason

    return False, None


def detect_anomaly(user, category: str, amount: float, expense_date: date, exclude_expense_id=None):
    """
    Combines the global model and personal statistical check.
    Returns (is_anomaly: bool, reason: str or None).
    """
    personal_flag, personal_reason = _personal_anomaly_check(
        user, category, amount, exclude_expense_id=exclude_expense_id
    )
    if personal_flag:
        return True, personal_reason

    global_flag = _global_anomaly_check(category, amount, expense_date)
    if global_flag:
        return True, f"₹{amount:.0f} in {category} looks unusual compared with typical spending patterns."

    return False, None


# ---------------------------------------------------------------------------
# Helper: monthly totals / budget calculations
# ---------------------------------------------------------------------------
def get_monthly_summary(user: User, year: int, month: int):
    total = (
        db.session.query(func.coalesce(func.sum(Expense.amount), 0.0))
        .filter(
            Expense.user_id == user.id,
            extract("year", Expense.date) == year,
            extract("month", Expense.date) == month,
        )
        .scalar()
    )
    count = (
        db.session.query(func.count(Expense.id))
        .filter(
            Expense.user_id == user.id,
            extract("year", Expense.date) == year,
            extract("month", Expense.date) == month,
        )
        .scalar()
    )

    budget = user.monthly_budget or 0.0
    remaining = budget - total
    if budget > 0:
        percent_used = min(100.0, max(0.0, (total / budget) * 100))
    else:
        percent_used = 0.0

    return {
        "total": round(total, 2),
        "count": count,
        "budget": round(budget, 2),
        "remaining": round(remaining, 2),
        "percent_used": round(percent_used, 1),
        "exceeded": total > budget and budget > 0,
        "approaching_threshold": (
            budget > 0 and percent_used >= (user.budget_threshold or 80)
        ),
    }


# ---------------------------------------------------------------------------
# Transaction import (CSV upload or pasted text)
#
# Solves the "cold start" problem: a brand-new user's dashboard has nothing
# useful on it until enough expenses exist for the anomaly detector and
# forecaster to say anything. Importing existing transactions fills that
# history in one step.
#
# Accepted formats (header row optional, order flexible if a header is given):
#     date, description, amount, category      <- category optional
#     2026-09-01, Lunch at cafe, 250, Food
#     2026-09-02, Metro card, 80
#
# Any row without a category is classified by the SAME ML model used on the
# Add Expense page, so imported rows are categorized automatically.
# ---------------------------------------------------------------------------
import csv as _csv
import io as _io

# Column header aliases, so common bank/wallet exports work without editing.
_DATE_ALIASES = {"date", "transaction date", "txn date", "posted date", "time"}
_DESC_ALIASES = {"description", "note", "narration", "details", "particulars",
                 "remarks", "transaction details", "memo", "name"}
_AMOUNT_ALIASES = {"amount", "inr", "debit", "value", "amt", "withdrawal",
                   "transaction amount"}
_CATEGORY_ALIASES = {"category", "type", "tag"}

_DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d.%m.%Y",
    "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M", "%m/%d/%Y %H:%M",
]


def _parse_date_flexible(raw):
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    # Last resort: let pandas try, it handles many odd formats.
    try:
        import pandas as pd
        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        if parsed is not None and not pd.isna(parsed):
            return parsed.date()
    except Exception:
        pass
    return None


def _parse_amount_flexible(raw):
    """Handles '1,250.50', '₹250', '250.00 Dr', '-250' etc."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Strip currency symbols, commas, and common debit/credit suffixes.
    for junk in ["₹", "Rs.", "Rs", "INR", ",", "Dr", "DR", "dr", "Cr", "CR", "cr"]:
        text = text.replace(junk, "")
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        return None
    # Bank exports often show debits as negative - treat magnitude as the expense.
    value = abs(value)
    if value <= 0:
        return None
    return value


def _detect_columns(header_row):
    """
    Given a header row, return {field: index} for whichever of
    date/description/amount/category we can identify. Returns {} if the
    row doesn't look like a header at all.
    """
    mapping = {}
    for idx, cell in enumerate(header_row):
        name = (cell or "").strip().lower()
        if name in _DATE_ALIASES and "date" not in mapping:
            mapping["date"] = idx
        elif name in _DESC_ALIASES and "description" not in mapping:
            mapping["description"] = idx
        elif name in _AMOUNT_ALIASES and "amount" not in mapping:
            mapping["amount"] = idx
        elif name in _CATEGORY_ALIASES and "category" not in mapping:
            mapping["category"] = idx
    # Only treat as a header if we found at least description + amount.
    if "description" in mapping and "amount" in mapping:
        return mapping
    return {}


def parse_transactions(raw_text):
    """
    Parse CSV/pasted transaction text into a list of dicts:
        {"date":, "description":, "amount":, "category":, "ai_categorized":}
    Returns (rows, errors) where errors is a list of human-readable strings.
    Never raises on bad input - malformed rows are skipped and reported.
    """
    rows = []
    errors = []

    if not raw_text or not raw_text.strip():
        return rows, ["No data provided."]

    # Sniff the delimiter (comma, tab, semicolon) from the first real line.
    sample = raw_text.strip().splitlines()[0]
    delimiter = ","
    for candidate in ["\t", ";", ","]:
        if candidate in sample:
            delimiter = candidate
            break

    try:
        reader = list(_csv.reader(_io.StringIO(raw_text), delimiter=delimiter))
    except Exception as e:
        return rows, [f"Could not read the data: {e}"]

    reader = [r for r in reader if any((c or "").strip() for c in r)]
    if not reader:
        return rows, ["No usable rows found."]

    # Try to detect a header row.
    column_map = _detect_columns(reader[0])
    if column_map:
        data_rows = reader[1:]
    else:
        # No header - assume positional: date, description, amount, [category]
        column_map = {"date": 0, "description": 1, "amount": 2}
        if len(reader[0]) >= 4:
            column_map["category"] = 3
        data_rows = reader

    valid_categories = set(app.config["CATEGORIES"])

    for line_no, row in enumerate(data_rows, start=2 if column_map else 1):
        def cell(field):
            idx = column_map.get(field)
            if idx is None or idx >= len(row):
                return ""
            return (row[idx] or "").strip()

        description = cell("description")
        amount = _parse_amount_flexible(cell("amount"))
        expense_date = _parse_date_flexible(cell("date"))
        category_raw = cell("category")

        if not description:
            errors.append(f"Row {line_no}: missing description - skipped.")
            continue
        if amount is None:
            errors.append(
                f"Row {line_no} ('{description[:30]}'): invalid or missing amount - skipped."
            )
            continue
        if expense_date is None:
            # Date is the most forgiving field - default to today rather than
            # throwing away an otherwise-valid transaction.
            expense_date = date.today()

        # Match category case-insensitively; anything unrecognized falls
        # through to the ML classifier rather than being silently wrong.
        ai_categorized = False
        category = None
        if category_raw:
            for valid in valid_categories:
                if category_raw.lower() == valid.lower():
                    category = valid
                    break
        if category is None:
            category = predict_category(description)
            ai_categorized = True

        rows.append({
            "date": expense_date,
            "description": description[:255],
            "amount": amount,
            "category": category,
            "ai_categorized": ai_categorized,
        })

    if not rows and not errors:
        errors.append("No valid transactions found.")

    return rows, errors


# ---------------------------------------------------------------------------
# Routes: Home / Auth
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("register.html")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("An account with this email already exists.", "error")
            return render_template("register.html")

        new_user = User(
            name=name,
            email=email,
            password=generate_password_hash(password),
            monthly_budget=0.0,
            budget_threshold=80,
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session["user_id"] = user.id
        session["user_name"] = user.name

        flash("Login successful.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes: Dashboard
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    today = date.today()
    summary = get_monthly_summary(user, today.year, today.month)

    recent_expenses = (
        Expense.query.filter_by(user_id=user.id)
        .order_by(Expense.date.desc(), Expense.created_at.desc())
        .limit(5)
        .all()
    )

    forecast = forecast_monthly_spending(user)
    insights = build_smart_insights(user, summary, forecast)

    return render_template(
        "dashboard.html",
        user=user,
        summary=summary,
        recent_expenses=recent_expenses,
        insights=insights,
        forecast=forecast,
    )


def build_smart_insights(user, summary, forecast=None):
    """
    Combines simple database calculations with budget info.
    ML-based insights (forecast, anomaly) are added in later phases.
    """
    insights = []

    if summary["budget"] > 0:
        insights.append({
            "type": "calculated",
            "icon": "chart",
            "text": f"You have used {summary['percent_used']}% of your monthly budget.",
        })

    if summary["exceeded"]:
        insights.append({
            "type": "calculated",
            "icon": "warning",
            "text": "Monthly budget exceeded.",
        })
    elif summary["approaching_threshold"]:
        insights.append({
            "type": "calculated",
            "icon": "warning",
            "text": "You are approaching your monthly budget.",
        })

    if summary["count"] == 0:
        insights.append({
            "type": "calculated",
            "icon": "info",
            "text": "No expenses recorded yet this month.",
        })

    today = date.today()
    anomalous_this_month = (
        Expense.query.filter(
            Expense.user_id == user.id,
            Expense.is_anomaly.is_(True),
            extract("year", Expense.date) == today.year,
            extract("month", Expense.date) == today.month,
        ).all()
    )
    anomaly_count = len(anomalous_this_month)
    if anomaly_count > 0:
        anomaly_total = sum(e.amount for e in anomalous_this_month)
        plural = "expenses" if anomaly_count > 1 else "expense"
        insights.append({
            "type": "ml_anomaly",
            "icon": "anomaly",
            "text": (
                f"🚨 {anomaly_count} unusual {plural} detected this month, "
                f"totaling ₹{anomaly_total:,.2f}."
            ),
        })

    if forecast and forecast.get("available"):
        predicted = forecast["predicted"]
        insights.append({
            "type": "ml_forecast",
            "icon": "forecast",
            "text": f"📈 Your projected monthly spending is ₹{predicted:,.0f}.",
        })

        if summary["budget"] > 0 and predicted > summary["budget"]:
            overage = predicted - summary["budget"]
            insights.append({
                "type": "ml_forecast",
                "icon": "warning",
                "text": f"⚠ You may exceed your budget by approximately ₹{overage:,.0f}.",
            })

    return insights


# ---------------------------------------------------------------------------
# Routes: Expenses (Add / List / Edit / Delete)
# ---------------------------------------------------------------------------
@app.route("/expenses/add", methods=["GET", "POST"])
@login_required
def add_expense():
    user = current_user()

    if request.method == "POST":
        description = request.form.get("description", "").strip()
        amount_raw = request.form.get("amount", "").strip()
        category = request.form.get("category", "Other").strip()
        date_raw = request.form.get("date", "").strip()

        if not description:
            flash("Description cannot be empty.", "error")
            return render_template("add_expense.html", categories=app.config["CATEGORIES"])

        try:
            amount = float(amount_raw)
            if amount <= 0:
                raise ValueError
        except ValueError:
            flash("Please enter a valid amount.", "error")
            return render_template("add_expense.html", categories=app.config["CATEGORIES"])

        try:
            expense_date = datetime.strptime(date_raw, "%Y-%m-%d").date()
        except ValueError:
            expense_date = date.today()

        if category not in app.config["CATEGORIES"]:
            category = "Other"

        is_anomaly, anomaly_reason = detect_anomaly(user, category, amount, expense_date)

        expense = Expense(
            user_id=user.id,
            amount=amount,
            description=description,
            category=category,
            date=expense_date,
            is_anomaly=is_anomaly,
            anomaly_reason=anomaly_reason,
        )
        db.session.add(expense)
        db.session.commit()

        flash("Expense added successfully.", "success")
        if is_anomaly:
            flash(f"🚨 Unusual Expense: {anomaly_reason}", "warning")
        return redirect(url_for("expenses"))

    return render_template(
        "add_expense.html",
        categories=app.config["CATEGORIES"],
        today=date.today().isoformat(),
    )


@app.route("/expenses/import", methods=["GET", "POST"])
@login_required
def import_expenses():
    """
    Bulk-import existing transactions from a CSV file or pasted text.
    Rows without a category are classified by the ML model automatically.
    """
    user = current_user()

    if request.method == "POST":
        raw_text = ""

        uploaded = request.files.get("csv_file")
        if uploaded and uploaded.filename:
            try:
                raw_bytes = uploaded.read()
                # Try UTF-8 first, fall back to latin-1 so odd exports still load.
                try:
                    raw_text = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    raw_text = raw_bytes.decode("latin-1", errors="replace")
            except Exception as e:
                print(f"[ExpenseAI] Could not read uploaded file: {e}")
                flash("Could not read that file. Please check it and try again.", "error")
                return redirect(url_for("import_expenses"))

        # Pasted text is used if no file was uploaded.
        if not raw_text.strip():
            raw_text = request.form.get("pasted_text", "")

        if not raw_text.strip():
            flash("Please upload a CSV file or paste some transactions.", "error")
            return redirect(url_for("import_expenses"))

        rows, errors = parse_transactions(raw_text)

        if not rows:
            flash("No valid transactions could be imported.", "error")
            for err in errors[:5]:
                flash(err, "error")
            return redirect(url_for("import_expenses"))

        # Insert the parsed rows. Anomaly detection runs per row, and because
        # each insert becomes part of the user's history, later rows are
        # checked against the history built up by earlier ones.
        imported = 0
        ai_categorized = 0
        anomalies = 0

        for row in rows:
            is_anomaly, anomaly_reason = detect_anomaly(
                user, row["category"], row["amount"], row["date"]
            )
            expense = Expense(
                user_id=user.id,
                amount=row["amount"],
                description=row["description"],
                category=row["category"],
                date=row["date"],
                is_anomaly=is_anomaly,
                anomaly_reason=anomaly_reason,
            )
            db.session.add(expense)
            db.session.flush()  # so the next row's history check sees this one
            imported += 1
            if row["ai_categorized"]:
                ai_categorized += 1
            if is_anomaly:
                anomalies += 1

        db.session.commit()

        flash(f"Imported {imported} transaction{'s' if imported != 1 else ''} successfully.", "success")
        if ai_categorized:
            flash(
                f"{ai_categorized} transaction{'s were' if ai_categorized != 1 else ' was'} "
                "categorized automatically by the AI model.",
                "success",
            )
        if anomalies:
            flash(
                f"🚨 {anomalies} unusual expense{'s' if anomalies != 1 else ''} detected "
                "in the imported data.",
                "warning",
            )
        if errors:
            flash(
                f"{len(errors)} row{'s were' if len(errors) != 1 else ' was'} skipped.",
                "error",
            )
            for err in errors[:3]:
                flash(err, "error")

        return redirect(url_for("expenses"))

    return render_template("import_expenses.html", categories=app.config["CATEGORIES"])


@app.route("/expenses")
@login_required
def expenses():
    user = current_user()
    all_expenses = (
        Expense.query.filter_by(user_id=user.id)
        .order_by(Expense.date.desc(), Expense.created_at.desc())
        .all()
    )
    return render_template("expenses.html", expenses=all_expenses)


@app.route("/expenses/export")
@login_required
def export_expenses():
    """
    Downloads all of the logged-in user's expenses as a CSV file.
    Pairs with the import feature - data can go in and come back out.
    """
    user = current_user()
    all_expenses = (
        Expense.query.filter_by(user_id=user.id)
        .order_by(Expense.date.desc(), Expense.created_at.desc())
        .all()
    )

    output = _io.StringIO()
    writer = _csv.writer(output)
    writer.writerow(["Date", "Description", "Category", "Amount", "Unusual", "Reason"])
    for exp in all_expenses:
        writer.writerow([
            exp.date.isoformat(),
            exp.description,
            exp.category,
            f"{exp.amount:.2f}",
            "Yes" if exp.is_anomaly else "No",
            exp.anomaly_reason or "",
        ])

    csv_data = output.getvalue()
    filename = f"expenseai_export_{date.today().isoformat()}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/expenses/edit/<int:expense_id>", methods=["GET", "POST"])
@login_required
def edit_expense(expense_id):
    user = current_user()
    expense = Expense.query.filter_by(id=expense_id, user_id=user.id).first()

    if not expense:
        flash("Expense not found.", "error")
        return redirect(url_for("expenses"))

    if request.method == "POST":
        description = request.form.get("description", "").strip()
        amount_raw = request.form.get("amount", "").strip()
        category = request.form.get("category", "Other").strip()
        date_raw = request.form.get("date", "").strip()

        if not description:
            flash("Description cannot be empty.", "error")
            return render_template(
                "edit_expense.html", expense=expense, categories=app.config["CATEGORIES"]
            )

        try:
            amount = float(amount_raw)
            if amount <= 0:
                raise ValueError
        except ValueError:
            flash("Please enter a valid amount.", "error")
            return render_template(
                "edit_expense.html", expense=expense, categories=app.config["CATEGORIES"]
            )

        try:
            expense_date = datetime.strptime(date_raw, "%Y-%m-%d").date()
        except ValueError:
            expense_date = expense.date

        if category not in app.config["CATEGORIES"]:
            category = "Other"

        is_anomaly, anomaly_reason = detect_anomaly(
            user, category, amount, expense_date, exclude_expense_id=expense.id
        )

        expense.description = description
        expense.amount = amount
        expense.category = category
        expense.date = expense_date
        expense.is_anomaly = is_anomaly
        expense.anomaly_reason = anomaly_reason

        db.session.commit()
        flash("Expense updated successfully.", "success")
        if is_anomaly:
            flash(f"🚨 Unusual Expense: {anomaly_reason}", "warning")
        return redirect(url_for("expenses"))

    return render_template(
        "edit_expense.html", expense=expense, categories=app.config["CATEGORIES"]
    )


@app.route("/expenses/delete/<int:expense_id>", methods=["POST"])
@login_required
def delete_expense(expense_id):
    user = current_user()
    expense = Expense.query.filter_by(id=expense_id, user_id=user.id).first()

    if not expense:
        flash("Expense not found.", "error")
        return redirect(url_for("expenses"))

    db.session.delete(expense)
    db.session.commit()
    flash("Expense deleted successfully.", "success")
    return redirect(url_for("expenses"))


# ---------------------------------------------------------------------------
# Route: ML category prediction API (used by Add Expense page JS)
# ---------------------------------------------------------------------------
@app.route("/predict-category", methods=["POST"])
@login_required
def predict_category_api():
    data = request.get_json(silent=True) or {}
    description = data.get("description", "")
    category = predict_category(description)
    return jsonify({"category": category})


# ---------------------------------------------------------------------------
# Route: Search (popup-based, current page context passed via ?next=)
# ---------------------------------------------------------------------------
@app.route("/search")
@login_required
def search():
    user = current_user()
    query = request.args.get("query", "").strip()

    results = []
    if query:
        like_pattern = f"%{query}%"
        results = (
            Expense.query.filter(
                Expense.user_id == user.id,
                db.or_(
                    Expense.description.ilike(like_pattern),
                    Expense.category.ilike(like_pattern),
                ),
            )
            .order_by(Expense.date.desc())
            .all()
        )

    return render_template("search.html", query=query, results=results)


# ---------------------------------------------------------------------------
# Route: Analytics
# ---------------------------------------------------------------------------
@app.route("/analytics")
@login_required
def analytics():
    user = current_user()
    all_expenses = Expense.query.filter_by(user_id=user.id).all()

    category_totals = {}
    for cat in app.config["CATEGORIES"]:
        category_totals[cat] = 0.0

    for exp in all_expenses:
        category_totals[exp.category] = category_totals.get(exp.category, 0.0) + exp.amount

    total_spent = sum(category_totals.values())

    highest_category = None
    highest_amount = 0.0
    for cat, amt in category_totals.items():
        if amt > highest_amount:
            highest_amount = amt
            highest_category = cat

    # Monthly trend for the last 6 months
    monthly_totals = get_monthly_trend(user, months=6)

    # Unusual spending summary - how much of the user's total spending was
    # flagged, not just how many transactions.
    anomalous_expenses = [e for e in all_expenses if e.is_anomaly]
    anomaly_count = len(anomalous_expenses)
    anomaly_total = round(sum(e.amount for e in anomalous_expenses), 2)
    anomaly_percent_of_spending = (
        round((anomaly_total / total_spent) * 100, 1) if total_spent > 0 else 0.0
    )
    # Most recent flagged expenses, for context on WHY they were flagged.
    recent_anomalies = sorted(
        anomalous_expenses, key=lambda e: (e.date, e.created_at), reverse=True
    )[:5]

    return render_template(
        "analytics.html",
        category_totals=category_totals,
        total_spent=round(total_spent, 2),
        highest_category=highest_category,
        highest_amount=round(highest_amount, 2),
        monthly_labels=[m["label"] for m in monthly_totals],
        monthly_values=[m["total"] for m in monthly_totals],
        has_data=len(all_expenses) > 0,
        anomaly_count=anomaly_count,
        anomaly_total=anomaly_total,
        anomaly_percent_of_spending=anomaly_percent_of_spending,
        recent_anomalies=recent_anomalies,
    )


def get_monthly_trend(user, months=6):
    today = date.today()
    results = []
    year, month = today.year, today.month

    for i in range(months - 1, -1, -1):
        m = month - i
        y = year
        while m <= 0:
            m += 12
            y -= 1

        total = (
            db.session.query(func.coalesce(func.sum(Expense.amount), 0.0))
            .filter(
                Expense.user_id == user.id,
                extract("year", Expense.date) == y,
                extract("month", Expense.date) == m,
            )
            .scalar()
        )

        label = date(y, m, 1).strftime("%b %Y")
        results.append({"label": label, "total": round(total, 2)})

    return results


# ---------------------------------------------------------------------------
# Phase 12 - Spending forecasting
#
# Combines two signals, each fit fresh for THIS user (a pretrained global
# model can't meaningfully predict another user's spending scale):
#   1. RUN-RATE: how much they've spent so far this month, projected out to
#      the full month (e.g. spent 60% of the month's days in and already at
#      60% of what they'd typically spend -> simple extrapolation).
#   2. TREND: a Linear Regression fitted on their own previous *complete*
#      months' totals, to capture a rising/falling spending trend over time.
#
# If enough history exists for both, they're averaged. If only one is
# available, that one is used alone. If neither is available, no prediction
# is invented - the spec is explicit about this.
# ---------------------------------------------------------------------------
MIN_DAYS_ELAPSED_FOR_RUN_RATE = 5
MIN_EXPENSES_FOR_RUN_RATE = 2
MIN_PREVIOUS_MONTHS_FOR_TREND = 2


def forecast_monthly_spending(user):
    """
    Returns a dict:
        {"available": True, "predicted": <float>, "methods": [...]}
    or
        {"available": False, "reason": "..."}
    """
    today = date.today()
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    day_of_month = today.day

    current_summary = get_monthly_summary(user, today.year, today.month)
    current_total = current_summary["total"]
    current_count = current_summary["count"]

    # Previous COMPLETE months only (this month is deliberately excluded).
    trend_data = get_monthly_trend(user, months=7)  # includes current month last
    previous_months = trend_data[:-1]
    previous_totals = [m["total"] for m in previous_months if m["total"] > 0]

    have_run_rate = (
        day_of_month >= MIN_DAYS_ELAPSED_FOR_RUN_RATE
        and current_count >= MIN_EXPENSES_FOR_RUN_RATE
    )
    have_trend = len(previous_totals) >= MIN_PREVIOUS_MONTHS_FOR_TREND

    if not have_run_rate and not have_trend:
        return {
            "available": False,
            "reason": "Not enough historical data for a reliable prediction.",
        }

    predictions = []
    methods = []

    if have_run_rate:
        run_rate_prediction = (current_total / day_of_month) * days_in_month
        predictions.append(run_rate_prediction)
        methods.append("current month run-rate")

    if have_trend:
        try:
            from sklearn.linear_model import LinearRegression

            X = np.arange(len(previous_totals)).reshape(-1, 1)
            y = np.array(previous_totals)
            model = LinearRegression()
            model.fit(X, y)
            next_index = np.array([[len(previous_totals)]])
            trend_prediction = max(0.0, float(model.predict(next_index)[0]))
            predictions.append(trend_prediction)
            methods.append("historical trend (linear regression)")
        except Exception as e:
            print(f"[ExpenseAI] Trend forecast failed, skipping: {e}")

    if not predictions:
        return {
            "available": False,
            "reason": "Not enough historical data for a reliable prediction.",
        }

    predicted = round(float(np.mean(predictions)), 2)

    return {"available": True, "predicted": predicted, "methods": methods}


# ---------------------------------------------------------------------------
# Route: Profile
# ---------------------------------------------------------------------------
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user()

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "personal":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()

            if not name or not email:
                flash("Name and email cannot be empty.", "error")
            else:
                existing = User.query.filter(
                    User.email == email, User.id != user.id
                ).first()
                if existing:
                    flash("That email is already in use.", "error")
                else:
                    user.name = name
                    user.email = email
                    db.session.commit()
                    session["user_name"] = user.name
                    flash("Profile updated successfully.", "success")

        elif form_type == "budget":
            budget_raw = request.form.get("monthly_budget", "").strip()
            threshold_raw = request.form.get("budget_threshold", "").strip()

            try:
                budget = float(budget_raw)
                if budget < 0:
                    raise ValueError
            except ValueError:
                flash("Budget must be a valid non-negative number.", "error")
                return redirect(url_for("profile"))

            try:
                threshold = int(threshold_raw)
                if not (0 <= threshold <= 100):
                    raise ValueError
            except ValueError:
                flash("Alert threshold must be between 0 and 100.", "error")
                return redirect(url_for("profile"))

            user.monthly_budget = budget
            user.budget_threshold = threshold
            db.session.commit()
            flash("Budget settings saved successfully.", "success")

        return redirect(url_for("profile"))

    return render_template("profile.html", user=user)


@app.route("/profile/password", methods=["POST"])
@login_required
def change_password():
    user = current_user()
    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not check_password_hash(user.password, current_password):
        flash("Current password is incorrect.", "error")
        return redirect(url_for("profile"))

    if new_password != confirm_password:
        flash("New passwords do not match.", "error")
        return redirect(url_for("profile"))

    if len(new_password) < 6:
        flash("New password must be at least 6 characters long.", "error")
        return redirect(url_for("profile"))

    user.password = generate_password_hash(new_password)
    db.session.commit()
    flash("Password changed successfully.", "success")
    return redirect(url_for("profile"))


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
def create_tables():
    # Ensure the database folder exists before SQLite tries to create the
    # .db file inside it - Git doesn't track empty directories, so if this
    # folder didn't survive the push to GitHub, SQLite would otherwise fail
    # with "unable to open database file" on a fresh deploy.
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
    os.makedirs(db_dir, exist_ok=True)

    with app.app_context():
        db.create_all()


# Always ensure tables exist when this module is imported - not just when run
# directly. A production server (gunicorn, etc.) imports `app` as a module
# and never executes the `if __name__ == "__main__"` block below, so table
# creation has to happen here instead, or the very first request would fail
# with "no such table: users".
create_tables()


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
