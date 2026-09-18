"""
ExpenseAI - Phase 6, Step 1 (v2 - expanded 8-category taxonomy)
Build a balanced ML training dataset from the user's original expense data.

Input:
    ml/expense_data_1.csv        (the user's original, untouched dataset)

Output:
    ml/expense_training.csv      (balanced: 40 rows per category, 320 total)

CATEGORY TAXONOMY CHANGE:
    The original build used 6 categories (Food, Transportation, Apparel,
    Household, Social Life, Other). This version expands to 8, matching
    everyday budgeting categories more people recognize:

        Food, Transport, Bills, Entertainment, Shopping, Health,
        Education, Other

    The raw dataset uses the old category names, so real historical rows
    are remapped via RAW_CATEGORY_MAP below:
        Food            -> Food
        Transportation  -> Transport
        Household       -> Bills          (rent, utilities, maintenance)
        Social Life     -> Entertainment  (movies, parties, outings)
        Apparel         -> Shopping       (clothing, accessories)
        Education       -> Education      (only 1 real example - heavily
                                            supplemented below)
        Other           -> Other
        Health          -> no real rows in this dataset at all; entirely
                            covered by curated supplemental examples.

Why supplementing is still needed:
    Several categories have very few or zero real examples after remapping
    (Bills: 6, Entertainment: 5, Shopping: 7, Education: 1, Health: 0).
    A classifier trained on such an imbalanced set would rarely predict the
    rare categories, so each category is topped up with a curated list of
    common, realistic expense descriptions until it reaches 40 rows.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "expense_data_1.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "expense_training.csv")

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

ROWS_PER_CATEGORY = 40

# For categories where the raw dataset has PLENTY of real examples (Food: 79,
# Other: 39), using real data alone actually hurts quality: much of it is
# noisy personal shorthand ("sent to vicky", "tea lights", "cycle gap")
# rather than clean, category-indicative text. Capping how many real
# examples get pulled in per category ensures the curated, clean
# SUPPLEMENTAL_EXAMPLES (deliberately written to be unambiguous) make up
# most of the training set, while still keeping some real grounding.
MAX_REAL_EXAMPLES_PER_CATEGORY = 10

# Maps the raw dataset's original category labels onto our new taxonomy.
# Any raw category not listed here (Allowance, Salary, Self-development,
# Beauty, Gift, Petty cash - each only 1 row) is dropped rather than
# force-fit into a category it doesn't clearly belong to.
RAW_CATEGORY_MAP = {
    "Food": "Food",
    "Transportation": "Transport",
    "Household": "Bills",
    "Social Life": "Entertainment",
    "Apparel": "Shopping",
    "Education": "Education",
    "Other": "Other",
}

# ---------------------------------------------------------------------------
# Curated supplemental examples, used to top up categories that don't have
# enough real examples in the raw dataset yet. These are realistic, everyday
# expense descriptions a person would actually type into the app.
# ---------------------------------------------------------------------------
SUPPLEMENTAL_EXAMPLES = {
    "Food": [
        "biryani", "pizza", "burger", "lunch", "dinner", "breakfast", "snacks",
        "shawarma", "parotta", "fried rice", "coffee", "tea", "ice cream",
        "juice", "restaurant bill", "food delivery", "swiggy order",
        "zomato order", "street food", "bakery items", "cake", "noodles",
        "sandwich", "chai and samosa", "buffet dinner", "fruits",
        "grocery snacks", "milkshake", "donuts", "chocolate", "canteen food",
        "mess bill", "food court", "chinese food", "south indian meal",
        "north indian thali", "momos", "rolls", "chaat", "cold drink",
        "groceries",
    ],
    "Transport": [
        "metro ticket", "bus ticket", "auto fare", "cab ride", "uber ride",
        "ola ride", "petrol", "diesel", "fuel refill", "train ticket",
        "flight ticket", "parking fee", "toll fee", "bike service",
        "car service", "vehicle repair", "rickshaw fare", "railway pass",
        "monthly bus pass", "app cab booking", "airport taxi", "fuel top up",
        "bike rental", "car rental", "ferry ticket", "shared cab",
        "highway toll", "vehicle insurance", "helmet purchase",
        "bike wash", "car wash", "public transport card recharge",
        "rapido ride", "shuttle service", "tempo travel", "scooter rental",
        "cycle repair", "commute fare", "travel booking fee", "cab surge fare",
    ],
    "Bills": [
        "rent", "electricity bill", "water bill", "gas cylinder",
        "wifi bill", "internet recharge", "mobile bill", "society maintenance",
        "house maintenance", "broadband bill", "dth recharge",
        "phone bill payment", "electricity recharge", "water tanker payment",
        "maintenance charges", "landlord payment", "utility bill",
        "gas bill", "cable tv bill", "postpaid bill", "prepaid recharge",
        "housing society fee", "property tax", "home loan emi",
        "apartment rent", "flat rent", "broadband recharge",
        "electric meter recharge", "piped gas bill", "garbage collection fee",
        "security deposit", "maintenance fee", "monthly rent payment",
        "cooking gas booking", "sewage bill", "municipal tax",
        "house rent advance", "utility recharge", "rent payment",
        "common area maintenance",
    ],
    "Entertainment": [
        "movie", "movie ticket", "concert ticket", "party", "club entry",
        "outing", "picnic", "bowling", "gaming zone", "amusement park",
        "night out", "bar bill", "pub visit", "get together",
        "festival celebration", "trip with friends", "karaoke night",
        "sports match ticket", "arcade games", "friends hangout",
        "cricket match tickets", "theatre play", "stand up comedy show",
        "reunion dinner", "farewell party", "new year party",
        "diwali celebration", "team outing", "friends trip", "cafe hangout",
        "clubbing", "date night", "celebration dinner", "netflix subscription",
        "spotify subscription", "video game purchase", "streaming subscription",
        "amusement ride tickets", "music concert", "live show tickets",
    ],
    "Shopping": [
        "shoes", "hoodie", "t-shirt", "jeans", "jacket", "shirt", "trousers",
        "sneakers", "sandals", "kurta", "dress", "skirt", "formal shirt",
        "winter jacket", "socks", "belt", "cap", "sunglasses", "watch",
        "handbag", "wallet purchase", "innerwear", "sweater", "blazer",
        "saree", "ethnic wear", "sports shoes", "slippers", "scarf",
        "gloves", "online shopping order", "amazon order", "flipkart order",
        "electronics purchase", "gadget purchase", "home decor shopping",
        "gift purchase", "accessories purchase", "mall shopping",
        "new phone case",
    ],
    "Health": [
        "doctor consultation", "pharmacy bill", "medicine purchase",
        "hospital visit", "dental checkup", "eye checkup", "health checkup",
        "gym membership", "yoga classes", "protein supplement",
        "medical test", "blood test", "vaccination", "physiotherapy session",
        "health insurance premium", "vitamins purchase", "first aid supplies",
        "clinic fee", "surgery cost", "lab test fee", "diagnostic center bill",
        "orthopedic consultation", "skin specialist visit", "dietitian consultation",
        "fitness tracker purchase", "medical checkup package",
        "eye glasses purchase", "contact lens purchase", "ambulance charges",
        "prescription refill", "therapy session", "mental health consultation",
        "covid test", "x-ray charges", "medical equipment purchase",
        "pain relief medicine", "cold and flu medicine", "dental filling",
        "root canal treatment", "health supplement subscription",
    ],
    "Education": [
        "college fees", "tuition fees", "school fees", "books purchase",
        "online course", "coaching class fee", "exam fee", "study material",
        "stationery items", "laptop for studies", "library fine",
        "certification course", "workshop fee", "seminar registration",
        "textbook purchase", "notebook purchase", "printing charges",
        "photocopy for assignment", "udemy course", "coursera subscription",
        "skill development course", "language class fee", "tutoring fee",
        "entrance exam fee", "application fee", "hostel fee",
        "project material cost", "lab fee", "student id card fee",
        "graduation fee", "convocation fee", "e-learning subscription",
        "competitive exam coaching", "spoken english class",
        "computer course fee", "music lessons", "dance class fee",
        "art class fee", "school uniform", "educational software license",
    ],
    "Other": [
        "mobile recharge", "document fee", "service payment",
        "repair payment", "miscellaneous payment", "other payment",
        "personal expense", "unexpected expense", "small expense",
        "general payment", "miscellaneous spending", "bank charges",
        "atm withdrawal fee", "late fee", "fine payment", "donation",
        "subscription renewal", "software subscription", "app purchase",
        "courier charges", "postage fee", "sim card purchase",
        "lended money", "borrowed money returned", "miscellaneous item",
        "service tax", "convenience fee", "processing fee",
        "loan repayment", "credit card payment", "membership fee",
        "annual maintenance charge", "unclassified expense",
        "cash withdrawal", "miscellaneous purchase", "gift for someone",
        "charity donation", "pet expense", "salon and grooming",
        "misc",
    ],
}


def clean_text(text: str) -> str:
    return str(text).strip().lower()


def build_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Could not find the original dataset at {RAW_DATA_PATH}. "
            "Place your expense_data_1.csv in the ml/ folder first."
        )

    raw_df = pd.read_csv(RAW_DATA_PATH)

    if "Note" not in raw_df.columns or "Category" not in raw_df.columns:
        raise ValueError(
            "expense_data_1.csv must contain 'Note' and 'Category' columns."
        )

    # Remap raw categories onto the new 8-category taxonomy; drop rows in
    # categories that don't map (Allowance, Salary, etc.) or with no note.
    raw_df = raw_df.copy()
    raw_df["MappedCategory"] = raw_df["Category"].map(RAW_CATEGORY_MAP)
    raw_df = raw_df.dropna(subset=["MappedCategory", "Note"])
    raw_df["Note"] = raw_df["Note"].apply(clean_text)
    raw_df = raw_df[raw_df["Note"] != ""]

    final_rows = []

    print("Building balanced training dataset (8-category taxonomy)...\n")

    for category in TARGET_CATEGORIES:
        real_examples = (
            raw_df[raw_df["MappedCategory"] == category]["Note"]
            .drop_duplicates()
            .tolist()
        )

        supplemental = [
            clean_text(x) for x in SUPPLEMENTAL_EXAMPLES.get(category, [])
        ]

        limited_real = real_examples[:MAX_REAL_EXAMPLES_PER_CATEGORY]

        # Curated examples go FIRST (clean, unambiguous, designed for this
        # category), then a limited number of real historical examples are
        # mixed in for authenticity. If that still isn't enough to reach the
        # quota, fall back to any remaining unused real examples.
        seen = set()
        combined = []
        for note in supplemental + limited_real:
            if note not in seen:
                seen.add(note)
                combined.append(note)

        if len(combined) < ROWS_PER_CATEGORY:
            for note in real_examples:
                if note not in seen:
                    seen.add(note)
                    combined.append(note)
                if len(combined) >= ROWS_PER_CATEGORY:
                    break

        if len(combined) < ROWS_PER_CATEGORY:
            print(
                f"  WARNING: Only {len(combined)} unique examples available "
                f"for '{category}' (need {ROWS_PER_CATEGORY}). "
                "Add more entries to SUPPLEMENTAL_EXAMPLES."
            )

        selected = combined[:ROWS_PER_CATEGORY]

        print(
            f"  {category:<15} -> {len(real_examples):>3} real available, "
            f"{min(len(real_examples), MAX_REAL_EXAMPLES_PER_CATEGORY):>2} used, "
            f"{len(selected):>3} total (target {ROWS_PER_CATEGORY})"
        )

        for note in selected:
            final_rows.append({"Note": note, "Category": category})

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved balanced training dataset to: {OUTPUT_PATH}")
    print(f"Final shape: {final_df.shape}")
    print("\nCategory counts:")
    print(final_df["Category"].value_counts())


if __name__ == "__main__":
    build_dataset()
