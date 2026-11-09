import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Merge tables
df = orders.merge(
    customers,
    on="customer_id"
)

df = df.merge(
    products,
    on="product_id"
)

completed = df[
    df["status"] == "completed"
]

print("=" * 50)
print("PIVOT TABLE")

pivot = pd.pivot_table(
    completed,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    margins=True
)

print(pivot)

print("\nGrand Total Revenue:")
print(round(
    pivot.loc["All", "All"],
    2
))

print("\n" + "=" * 50)
print("CITY AGGREGATION")

city_stats = (
    completed
    .groupby("city")
    .agg(
        revenue_sum=("order_amount", "sum"),
        revenue_mean=("order_amount", "mean"),
        order_count=("order_amount", "count")
    )
)

print(city_stats)

print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")

s = completed["order_amount"]

print("Mean:", round(s.mean(), 2))
print("Median:", round(s.median(), 2))
print("Skewness:", round(s.skew(), 2))

q1 = s.quantile(0.25)
q2 = s.quantile(0.50)
q3 = s.quantile(0.75)

print("\nQuartiles")
print("Q1:", round(q1, 2))
print("Q2:", round(q2, 2))
print("Q3:", round(q3, 2))

iqr = q3 - q1

lower_fence = q1 - (1.5 * iqr)
upper_fence = q3 + (1.5 * iqr)

print("\nIQR:", round(iqr, 2))
print("Lower Fence:", round(lower_fence, 2))
print("Upper Fence:", round(upper_fence, 2))

below = (s < lower_fence).sum()
above = (s > upper_fence).sum()

print("\nOutliers Below:", below)
print("Outliers Above:", above)

print("\n" + "=" * 50)
print("CORRELATION MATRIX")

corr_cols = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

corr = (
    completed[corr_cols]
    .corr()
)

print(corr)