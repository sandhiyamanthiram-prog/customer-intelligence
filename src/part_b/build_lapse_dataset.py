import pandas as pd

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

# ====================================
# Label Creation
# ====================================

completed = orders[
    orders["status"] == "completed"
]

cohort_customers = completed[
    completed["order_ts"] < cutoff
]["customer_id"].unique()

sept_orders = completed[
    (completed["order_ts"] >= "2025-09-01")
    &
    (completed["order_ts"] < "2025-10-01")
]

active_sept = set(
    sept_orders["customer_id"]
)

labels = pd.DataFrame({
    "customer_id": cohort_customers
})

labels["lapsed"] = (
    ~labels["customer_id"]
     .isin(active_sept)
).astype(int)

# ====================================
# Orders BEFORE cutoff ONLY
# ====================================

pre_cutoff = orders[
    orders["order_ts"] < cutoff
]

pre_completed = pre_cutoff[
    pre_cutoff["status"] == "completed"
]

# ====================================
# RFM Features
# ====================================

rfm = (
    pre_completed
    .groupby("customer_id")
    .agg(
        last_order=("order_ts","max"),
        frequency=("order_id","nunique"),
        monetary=("order_amount","sum"),
        avg_discount=("discount_pct","mean"),
        avg_delivery=("delivery_minutes","mean")
    )
)

rfm["recency"] = (
    cutoff - rfm["last_order"]
).dt.days

rfm = rfm.drop(
    columns=["last_order"]
)

# ====================================
# Cancellation Rate
# ====================================

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

# ====================================
# Favourite Category
# ====================================

tmp = pre_completed.merge(
    products[
        ["product_id","category"]
    ],
    on="product_id"
)

fav_cat = (
    tmp
    .groupby(
        ["customer_id","category"]
    )
    .size()
    .reset_index(
        name="cnt"
    )
)

fav_cat = (
    fav_cat
    .sort_values(
        ["customer_id","cnt"],
        ascending=[True,False]
    )
    .drop_duplicates(
        "customer_id"
    )
)

fav_cat = fav_cat[
    ["customer_id","category"]
]

fav_cat.columns = [
    "customer_id",
    "favorite_category"
]

# ====================================
# Merge Everything
# ====================================

dataset = (
    labels
    .merge(
        rfm,
        on="customer_id"
    )
    .merge(
        cancel_rate,
        on="customer_id",
        how="left"
    )
    .merge(
        fav_cat,
        on="customer_id",
        how="left"
    )
    .merge(
        customers,
        on="customer_id",
        how="left"
    )
)

dataset["tenure_days"] = (
    cutoff -
    dataset["signup_date"]
).dt.days

dataset = dataset.drop(
    columns=["signup_date"]
)

dataset["cancel_rate"] = (
    dataset["cancel_rate"]
    .fillna(0)
)

print(
    "Dataset Shape:",
    dataset.shape
)

print(
    "Customers:",
    len(dataset)
)

print(
    "Lapsed:",
    dataset["lapsed"].sum()
)

print(
    "Missing Values:"
)

print(
    dataset.isna().sum()
)

import os

os.makedirs(
    "data",
    exist_ok=True
)

dataset.to_csv(
    "data/lapse_dataset.csv",
    index=False
)

print(
    "\nDataset Saved"
)