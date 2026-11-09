import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

# =====================================
# Load Segments
# =====================================

segments = pd.read_csv(
    "outputs/customer_segments.csv"
)

# =====================================
# Rebuild Feature Dataset
# =====================================

data = pd.read_csv(
    "data/lapse_dataset.csv"
)

# Merge Premium Flag
data = data.merge(
    segments[
        ["customer_id", "is_premium"]
    ],
    on="customer_id",
    how="left"
)

# =====================================
# Train / Test Split
# =====================================

X = data.drop(
    columns=[
        "customer_id",
        "lapsed",
        "is_premium"
    ]
)

y = data["lapsed"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
)

# customer ids aligned with test set
test_customers = data.loc[
    X_test.index,
    [
        "customer_id",
        "is_premium"
    ]
]

# =====================================
# Load Model
# =====================================

model = joblib.load(
    "outputs/lapse_model.joblib"
)

# =====================================
# Probabilities
# =====================================

probs = model.predict_proba(
    X_test
)[:, 1]

results = pd.DataFrame({
    "customer_id":
        test_customers["customer_id"].values,

    "is_premium":
        test_customers["is_premium"].values,

    "lapse_probability":
        probs,

    "actually_lapsed":
        y_test.values
})

# =====================================
# Premium Customers Only
# =====================================

premium = results[
    results["is_premium"] == True
]

premium = (
    premium
    .sort_values(
        "lapse_probability",
        ascending=False
    )
    .head(10)
)

# =====================================
# Save
# =====================================

os.makedirs(
    "outputs",
    exist_ok=True
)

premium[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
].to_csv(
    "outputs/retention_list.csv",
    index=False
)

print(
    "\nRetention List Saved"
)

print(
    premium[
        [
            "customer_id",
            "lapse_probability",
            "actually_lapsed"
        ]
    ]
)

print(
    "\nActually Lapsed:",
    premium[
        "actually_lapsed"
    ].sum()
)