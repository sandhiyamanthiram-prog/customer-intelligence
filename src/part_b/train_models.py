import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier


# ==========================
# Load Feature Dataset
# ==========================

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

cutoff = pd.Timestamp("2025-09-01")

completed = orders[
    orders["status"] == "completed"
]

cohort = completed[
    completed["order_ts"] < cutoff
]["customer_id"].unique()

sept = completed[
    (completed["order_ts"] >= "2025-09-01")
    &
    (completed["order_ts"] < "2025-10-01")
]

active = set(
    sept["customer_id"]
)

labels = pd.DataFrame({
    "customer_id": cohort
})

labels["lapsed"] = (
    ~labels["customer_id"]
    .isin(active)
).astype(int)

# ==========================
# Build Features
# ==========================

pre_cutoff = orders[
    orders["order_ts"] < cutoff
]

pre_completed = pre_cutoff[
    pre_cutoff["status"] == "completed"
]

rfm = (
    pre_completed
    .groupby("customer_id")
    .agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean")
    )
)

rfm["recency"] = (
    cutoff - rfm["last_order"]
).dt.days

rfm = rfm.drop(
    columns=["last_order"]
)

order_counts = (
    pre_cutoff
    .groupby("customer_id")
    .size()
)

cancel_counts = (
    pre_cutoff[
        pre_cutoff["status"]=="cancelled"
    ]
    .groupby("customer_id")
    .size()
)

cancel_rate = (
    cancel_counts
    .div(order_counts)
    .fillna(0)
)

cancel_rate.name = "cancel_rate"

tmp = pre_completed.merge(
    products[
        ["product_id","category"]
    ],
    on="product_id"
)

fav = (
    tmp.groupby(
        ["customer_id","category"]
    )
    .size()
    .reset_index(name="cnt")
)

fav = (
    fav
    .sort_values(
        ["customer_id","cnt"],
        ascending=[True,False]
    )
    .drop_duplicates("customer_id")
)

fav = fav[
    ["customer_id","category"]
]

fav.columns = [
    "customer_id",
    "favorite_category"
]

data = (
    labels
    .merge(rfm,on="customer_id")
    .merge(cancel_rate,
           on="customer_id",
           how="left")
    .merge(fav,
           on="customer_id",
           how="left")
    .merge(customers,
           on="customer_id",
           how="left")
)

data["tenure_days"] = (
    cutoff - data["signup_date"]
).dt.days

data = data.drop(
    columns=["signup_date"]
)

# ==========================
# Train / Test Split
# ==========================

X = data.drop(
    columns=["customer_id","lapsed"]
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

print(
    "Train:",
    len(X_train)
)

print(
    "Test:",
    len(X_test)
)

# ==========================
# Baseline
# ==========================

baseline = [0]*len(y_test)

print("\nBaseline")

print(
    "F1:",
    f1_score(
        y_test,
        baseline,
        zero_division=0
    )
)

# ==========================
# Preprocessing
# ==========================

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

# ==========================
# Logistic Regression
# ==========================

log_pipe = Pipeline(
    [
        ("prep", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

log_grid = GridSearchCV(
    log_pipe,
    {
        "model__C":[
            0.01,
            0.1,
            1,
            10,
            100
        ]
    },
    cv=5,
    scoring="f1"
)

log_grid.fit(
    X_train,
    y_train
)

print("\nBest Logistic")

print(
    log_grid.best_params_
)

print(
    log_grid.best_score_
)

# ==========================
# Random Forest
# ==========================

rf_pipe = Pipeline(
    [
        ("prep", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

rf_grid = GridSearchCV(
    rf_pipe,
    {
        "model__max_depth":[3,5,8],
        "model__min_samples_leaf":[5,10,20]
    },
    cv=5,
    scoring="f1"
)

rf_grid.fit(
    X_train,
    y_train
)

print("\nBest RF")

print(
    rf_grid.best_params_
)

print(
    rf_grid.best_score_
)

best_model = rf_grid

pred = best_model.predict(
    X_test
)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        pred
    )
)

print(
    "\nPrecision:",
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

os.makedirs(
    "outputs",
    exist_ok=True
)

joblib.dump(
    best_model,
    "outputs/lapse_model.joblib"
)

print(
    "\nModel Saved"
)

# =====================================
# Reload Verification
# =====================================

loaded_model = joblib.load(
    "outputs/lapse_model.joblib"
)

pred_original = best_model.predict(
    X_test
)

pred_loaded = loaded_model.predict(
    X_test
)

print(
    "\nPredictions Identical:",
    (pred_original == pred_loaded).all()
)

best_rf = rf_grid.best_estimator_

rf_model = best_rf.named_steps["model"]

feature_names = (
    best_rf.named_steps["prep"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)

