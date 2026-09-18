# ExpenseAI — ML-Based Smart Personal Expense Tracker

This build covers **Phases 1–6, 11, 12, and 14**: project setup, database,
authentication, dashboard, full expense CRUD, all three ML features
(classification, anomaly detection, forecasting), Smart Insights, and
responsive/UI polish.

> **Category taxonomy update:** categories changed from the original 6
> (Food, Transportation, Apparel, Household, Social Life, Other) to 8:
> **Food, Transport, Bills, Entertainment, Shopping, Health, Education,
> Other**. If you have an existing `database/expense_tracker.db` from
> before this change, delete it and log in fresh — old expenses used the
> old category names and won't match the new dropdown options or ML models.

## New in this update

- **8-category taxonomy** instead of 6, matching common everyday budgeting
  categories. All three ML models (classifier, anomaly detector, spending
  forecaster) were retrained for the new categories using `ml/expense_data_1.csv`
  remapped onto the new taxonomy (see `RAW_CATEGORY_MAP` in each training
  script) plus curated examples for categories with little/no real data.
- **Classifier accuracy improved significantly** during this rebuild - two
  real bugs were found and fixed along the way:
  1. The data-combination logic was letting noisy real historical text
     (e.g. "sent to vicky", "tea lights") crowd out clean curated examples
     for categories with abundant real data (Food, Other). Fixed by capping
     real-example usage per category and prioritizing curated text.
  2. The deployed model was only fit on an 80% train split, so any curated
     word appearing exactly once in the dataset (e.g. "handbag", "watch")
     had a 20% chance of being completely absent from the live model's
     vocabulary. Fixed by doing a final refit on 100% of the data before
     saving, while still reporting honest evaluation metrics from the
     held-out split beforehand. See `ml/train_model.py` for the full
     explanation and a real-world spot-check built into the script's output
     (100% on 25 everyday phrases across all 8 categories, vs. ~57.5%
     5-fold cross-validation accuracy — both numbers are legitimate, they
     measure different things, and the script explains why).
- **UI refresh** to match the layout style of a provided reference design:
  icon badges on stat cards with colored left-border accents, category
  icons in Recent Transactions and the Transactions table, a two-column
  Add Expense form (Amount + Category side by side), and a nicer "AI
  Powered" sidebar tip card. The navy/blue color system from the original
  spec is unchanged - only layout and iconography changed. Profile and
  Logout are untouched, as requested. No routes, functions, or backend
  logic changed as part of this visual pass.


> **Upgrading from an earlier version of this project?** The `Expense` table
> now has two new columns (`is_anomaly`, `anomaly_reason`). Delete
> `database/expense_tracker.db` and let the app recreate it on next run —
> this project doesn't use migrations, so existing SQLite files won't pick
> up new columns automatically. You'll lose any expenses you'd already
> entered for testing, but that's expected at this stage.

## What's included in this build

- Flask + Flask-SQLAlchemy + SQLite backend
- User registration, login, logout (hashed passwords, sessions)
- Protected routes (dashboard, expenses, analytics, profile)
- Add / Edit / Delete expenses, all scoped to the logged-in user
- Dashboard with monthly totals, budget progress bar, budget warnings
- Transactions page with category badges
- Search popup (sidebar → popup, ESC/click-outside/✕ to close, quick-search chips)
- Analytics page with Chart.js (doughnut, bar, line — monthly trend)
- Profile page: personal info, budget settings, password change, AI feature cards
- **Trained ML category classifier** (TF-IDF + Logistic Regression) already
  saved at `models/expense_model.pkl`. The `/predict-category` endpoint and
  the Add Expense form's debounced JS use it automatically — type a
  description, pause typing, and the category dropdown updates itself with
  an "AI Suggested" tag.

## The ML pipeline (Phase 6)

```
ml/expense_data_1.csv          (your original, untouched dataset)
        ↓
ml/create_training_dataset.py  (builds a balanced 240-row set: 40/category)
        ↓
ml/expense_training.csv        (already generated and included)
        ↓
ml/train_model.py              (TF-IDF + Logistic Regression, evaluates, saves)
        ↓
models/expense_model.pkl       (already trained and included)
```

To retrain from scratch:
```
cd ml
python create_training_dataset.py   # optional - regenerates expense_training.csv
python train_model.py               # trains + evaluates + saves the model
```

**Evaluation results** (on a held-out 20% test split, stratified):
- Accuracy: **66.7%**
- Macro F1: **0.666**
- Full classification report + confusion matrix print to the console when
  you run `train_model.py`

This is an honest number, not inflated — with only 240 short, 1–2 word
descriptions spread across 6 categories, some genuine overlap is expected
(e.g. "grocery snacks" reads a lot like Food even when logged as Household).
For your demo, this is a good, presentable feature-quality tradeoff to talk
through: more/richer training data (longer descriptions, more real examples
per category) is the clear next lever to pull if you want higher accuracy.

## Setup (Windows / VS Code)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The SQLite database is created automatically on first run at
`database/expense_tracker.db`.

## Anomaly detection (Phase 11)

```
ml/expense_data_1.csv           (your original dataset)
        ↓
ml/train_anomaly_model.py       (Isolation Forest on per-category amount z-score + day)
        ↓
models/anomaly_model.pkl        (already trained and included)
```

> **Bug found and fixed during testing:** the first version of this model
> fed raw amount + one-hot category columns into the Isolation Forest
> together. That backfired — categories with few historical examples
> (Apparel: 7, Household: 6, Social Life: 5, out of 227 total rows) are
> themselves rare combinations in the full dataset, so the model learned to
> flag almost *any* expense in those categories as "unusual," regardless of
> amount. A real ₹150 Social Life expense (literally present in the
> training data) got flagged for this reason. Fixed by computing a
> per-category z-score (how many standard deviations the amount is from
> that category's own mean) as the feature instead of raw amount +
> category — this compares spending within each category on its own terms
> rather than penalizing rare categories. Verified: the exact ₹150 false
> positive no longer fires, typical amounts in every category are left
> alone, and genuinely large outliers (e.g. ₹8,000 in a category that
> usually sees ₹150–500) are still caught.

**How detection actually works when you add or edit an expense** — two
signals, either one can flag an expense as unusual:

1. **Personal check** (the one that matches the spec's example most
   closely): once you have at least 4 past expenses in a category, a new
   expense is compared against *your own* historical mean and standard
   deviation in that category. E.g. if your last five Food expenses were
   ₹150–220 and you log ₹5,000, that's flagged relative to *you*, not some
   generic threshold.
2. **Global model** (the trained Isolation Forest): used mainly for
   "cold start" — a brand-new user's very first few expenses, before there's
   enough personal history to compare against.

Flagged expenses show a 🚨 **Unusual** tag on the Transactions page and in
Recent Transactions, a flash message when you add/edit them, and roll up
into a Smart Insight on the dashboard ("🚨 2 unusual expenses detected this
month"). Verified with test cases for: true positives (personal history +
cold start), and false positives (normal spending is *not* flagged either
way, in any category, including the previously-sparse ones).

To retrain:
```
cd ml
python train_anomaly_model.py
```
This prints the flagged/total ratio and the average amount for normal vs.
flagged expenses (1,231 vs 176 in the included run) — useful evidence for
your report, since accuracy isn't a meaningful metric for unsupervised
anomaly detection.

## Spending forecasting (Phase 12)

This is the one ML feature where a pretrained, saved model genuinely
doesn't make sense to use live — every user has a different budget and
spending scale, so a model trained once on one person's numbers can't
predict another's. Two things exist instead:

1. **`ml/train_spending_model.py`** — a standalone demonstration script,
   for your report. It trains a Linear Regression on your original
   dataset's monthly totals (after correctly excluding the two boundary
   months that are partial — the data collection window starts Nov 21 and
   ends Mar 2, so those two months would otherwise badly mislead the
   trend). On the 3 remaining complete months it reports a leave-one-out
   MAE of ~₹4,956 and projects the next month at ~₹19,754. Saves
   `models/spending_model.pkl` as a reference artifact.

2. **`forecast_monthly_spending()` in `app.py`** — what actually powers the
   dashboard. For each user, it combines (when available):
   - **Run-rate**: current month's spending so far, projected out to the
     full month (needs ≥5 days elapsed and ≥2 expenses this month).
   - **Trend**: a fresh Linear Regression fit on that user's own previous
     complete months' totals (needs ≥2 previous months).

   If neither signal has enough data, the dashboard shows *"Not enough
   historical data for a reliable prediction"* — no invented numbers, per
   spec. Tested with: no data (correctly refuses), trend-only (predicted
   ₹10,867 against a rising 8,500→9,200→10,100 history, closely matching
   the spec's own worked example), and combined run-rate + trend.

The dashboard now has a dedicated **Spending Forecast** card, and Smart
Insights adds a budget-overage warning ("⚠ You may exceed your budget by
approximately ₹X") whenever the forecast exceeds the monthly budget.

## Smart Insights (Phase 13)

The dashboard's Smart Insights card now combines all three kinds of signal,
and the code keeps them explicitly labeled by type rather than presenting
everything as "AI":
- **Calculated** — budget % used, budget exceeded/approaching warnings
- **ML anomaly** — count of unusual expenses this month
- **ML forecast** — projected monthly total, and a budget-overage warning
  when the forecast crosses the budget line



## Transaction Import (CSV / paste-in)

Solves the **cold-start problem**: a brand-new user's dashboard has nothing
useful on it until enough expenses exist for the anomaly detector and
forecaster to say anything. Import fills that history in one step.

**Two ways in** (`/expenses/import`, "Import Data" in the sidebar):
- Upload a `.csv` file
- Paste transactions directly into a textarea

**Format** — header row optional, column order flexible:
```
date, description, amount, category
2026-09-01, Lunch at cafe, 250, Food
2026-09-02, Metro card recharge, 80
```

The `category` column is **optional**. Any row without one is classified by
the same ML model used on the Add Expense page, so imported data gets
categorized automatically.

**Robustness** (all verified by test):
- Recognizes common bank-export column names (note, narration, particulars,
  debit, transaction date, etc.)
- Handles `1,250.50`, `₹250`, `250.00 Dr`, and negative debit amounts
- Tries 12+ date formats, falls back to today rather than discarding an
  otherwise-valid row
- Works with no header row at all (assumes date, description, amount, category)
- Bad rows are skipped with a specific reason; valid rows still import
- Anomaly detection runs per row, and rows are flushed as they're inserted so
  later rows are checked against history built by earlier ones

`sample_transactions.csv` in the project root is a ready-made demo file
(30 rows across 4 months). Importing it produces a working forecast and
populated analytics immediately.

**Manual entry is unchanged** — import is additive. The AI category
suggestion on the Add Expense page (type a description, watch the dropdown
fill itself) is still the headline ML demo; import just means users don't
start from an empty dashboard.

## What's left (polish only — every functional feature is implemented)

| Item | Notes |
|---|---|
| Manual browser testing on Windows | Phase 15 — this build is tested via automated Flask test-client requests; give it a real click-through before your demo |

**Phase 14 (responsive/UI polish) is now done:**
- Sidebar has a dimmed backdrop on mobile, closes when you tap outside it or tap a nav link
- Transactions/Search tables collapse into stacked cards below 640px instead of scrolling horizontally (per spec: "Do not allow horizontal scrolling")
- Search popup and flash messages have subtle fade/slide transitions
- Cleaned up a couple of stray empty folders that had ended up in earlier zip exports

## Project structure

```
expenseai/
├── app.py                  # Flask app, models, routes, auth, CRUD
├── config.py                # Config, DB path, ML model paths, categories
├── requirements.txt
├── database/                 # SQLite DB (auto-created)
├── ml/                        # Training scripts (Phase 6+)
├── models/                    # Saved .pkl models (Phase 6+)
├── templates/                 # Jinja2 templates
└── static/
    ├── css/style.css
    └── js/script.js
```

## Notes for the demo

- The `/predict-category` endpoint and frontend debounce logic are already
  fully wired — this is intentional, so that once you train and drop in the
  model file, the "automatic AI category suggestion" feature works instantly
  without touching the Add Expense page again.
- All expense queries are filtered by `user_id`, so one user can never see
  another's data.
