import pandas as pd

orders = pd.read_csv(
    "data/clean/orders.csv"
)

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

completed = orders[
    orders["status"] == "completed"
]

snapshot = pd.Timestamp(
    "2025-10-01 00:00:00"
)

rfm = completed.groupby(
    "customer_id"
).agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
)

rfm["recency"] = (
    snapshot - rfm["last_order"]
).dt.days

rfm = rfm.reset_index()

print("Customers:", len(rfm))

print("\nRecency")
print(rfm["recency"].describe())

print("\nFrequency")
print(rfm["frequency"].describe())

print("\nMonetary")
print(rfm["monetary"].describe())