import pandas as pd
import os

# =========================
# Create clean folder
# =========================
os.makedirs("data/clean", exist_ok=True)

# =========================
# Read raw files
# =========================
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

print("Customers Shape:", customers.shape)
print("Products Shape:", products.shape)
print("Orders Shape:", orders.shape)

# =========================
# Missing Values Audit
# =========================
print("\nMissing Values - Customers")
print(customers.isna().sum())

print("\nMissing Values - Products")
print(products.isna().sum())

print("\nMissing Values - Orders")
print(orders.isna().sum())

# ==================================================
# CUSTOMER CLEANING
# ==================================================

# Standardize city names
customers["city"] = (
    customers["city"]
    .astype(str)
    .str.strip()
    .str.title()
)

print("\nCities:")
print(customers["city"].value_counts())

# Standardize gender
gender_map = {
    "male": "male",
    "Male": "male",
    "M": "male",
    "m": "male",
    "MALE": "male",

    "female": "female",
    "Female": "female",
    "F": "female",
    "f": "female",
    "FEMALE": "female"
}

customers["gender"] = customers["gender"].replace(gender_map)
customers["gender"] = customers["gender"].replace("", "unknown")
customers["gender"] = customers["gender"].fillna("unknown")

print("\nGender Counts:")
print(customers["gender"].value_counts())

# Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

# Treat age outside 10-100 as missing
customers.loc[
    (customers["age"] < 10) |
    (customers["age"] > 100),
    "age"
] = pd.NA

# Fill missing ages with city median
customers["age"] = (
    customers.groupby("city")["age"]
    .transform(lambda x: x.fillna(x.median()))
)

customers["age"] = customers["age"].astype(int)

# Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

print("\nMissing Ages:")
print(customers["age"].isna().sum())

# Save customers
customers.to_csv(
    "data/clean/customers.csv",
    index=False
)

print("\ncustomers.csv saved successfully")

# ==================================================
# PRODUCTS
# ==================================================

products.to_csv(
    "data/clean/products.csv",
    index=False
)

print("products.csv saved successfully")

# ==================================================
# ORDER CLEANING
# ==================================================

# Convert timestamp
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

# Remove exact duplicate rows
orders = orders.drop_duplicates()

print("\nOrders after removing exact duplicates:")
print(len(orders))

# Keep latest row per order_id
orders = (
    orders
    .sort_values("order_ts")
    .drop_duplicates(
        subset="order_id",
        keep="last"
    )
)

print("\nUnique orders:")
print(len(orders))

# Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# Join product price
orders = orders.merge(
    products[
        ["product_id", "price_inr"]
    ],
    on="product_id",
    how="left"
)

# Recalculate missing amounts
missing_amt = orders["order_amount"].isna()

orders.loc[
    missing_amt,
    "order_amount"
] = (
    orders.loc[missing_amt, "price_inr"]
    * orders.loc[missing_amt, "quantity"]
    * (
        1 -
        orders.loc[missing_amt, "discount_pct"] / 100
    )
).round(2)

print(
    "\nMissing Amounts:",
    orders["order_amount"].isna().sum()
)

# ==================================================
# DELIVERY TIME CLEANING
# ==================================================

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# Negative delivery times -> missing
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

# Join category
orders = orders.merge(
    products[
        ["product_id", "category"]
    ],
    on="product_id",
    how="left"
)

completed = orders["status"] == "completed"

# Category medians
category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

# Fill missing delivery times only for completed orders
orders.loc[
    completed,
    "delivery_minutes"
] = (
    orders.loc[
        completed,
        "delivery_minutes"
    ].fillna(category_medians)
)

# Cancelled orders stay NULL
orders.loc[
    orders["status"] == "cancelled",
    "delivery_minutes"
] = pd.NA

# Remove helper columns
orders = orders.drop(
    columns=[
        "price_inr",
        "category"
    ]
)

# ==================================================
# CHECKPOINTS
# ==================================================

print("\nStatus Counts:")
print(
    orders["status"]
    .value_counts()
)

print("\nMissing Delivery By Status:")
print(
    orders.groupby("status")["delivery_minutes"]
    .apply(lambda x: x.isna().sum())
)

completed_revenue = round(
    orders.loc[
        orders["status"] == "completed",
        "order_amount"
    ].sum(),
    2
)

print("\nCompleted Revenue:")
print(completed_revenue)

# ==================================================
# SAVE CLEAN ORDERS
# ==================================================

orders.to_csv(
    "data/clean/orders.csv",
    index=False
)

print("\norders.csv saved successfully")