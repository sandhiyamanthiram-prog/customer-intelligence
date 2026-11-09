import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# ==================================
# Load Dataset
# ==================================

data = pd.read_csv(
    "data/lapse_dataset.csv"
)

# ==================================
# Features & Target
# ==================================

X = data.drop(
    columns=["customer_id", "lapsed"]
)

y = data["lapsed"]

# ==================================
# Split
# ==================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
)

# ==================================
# Show Original Class Counts
# ==================================

print("Original Training Classes")

print(
    y_train.value_counts()
)

# ==================================
# Oversampling
# ==================================

train_df = X_train.copy()

train_df["lapsed"] = y_train

majority = train_df[
    train_df["lapsed"] == 0
]

minority = train_df[
    train_df["lapsed"] == 1
]

minority_oversampled = (
    minority.sample(
        n=len(majority),
        replace=True,
        random_state=42
    )
)

balanced = pd.concat(
    [majority,
     minority_oversampled]
)

print("\nBalanced Classes")

print(
    balanced["lapsed"]
    .value_counts()
)

# ==================================
# Split Back
# ==================================

X_bal = balanced.drop(
    columns=["lapsed"]
)

y_bal = balanced["lapsed"]

# ==================================
# Preprocessing
# ==================================

num_cols = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "avg_delivery",
    "cancel_rate",
    "age",
    "tenure_days"
]

cat_cols = [
    "city",
    "gender",
    "favorite_category"
]

preprocessor = (
    ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                num_cols
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                cat_cols
            )
        ]
    )
)

# ==================================
# Logistic Regression
# ==================================

model = Pipeline(
    [
        ("prep", preprocessor),
        (
            "model",
            LogisticRegression(
                C=0.1,
                max_iter=1000
            )
        )
    ]
)

model.fit(
    X_bal,
    y_bal
)

pred = model.predict(
    X_test
)

print("\nAfter Balancing")

print(
    "Precision:",
    precision_score(
        y_test,
        pred
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        pred
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        pred
    )
)