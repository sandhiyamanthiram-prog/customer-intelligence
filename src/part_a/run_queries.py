import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect("data/instamart.db")

# ==================================================
# Q1 - WHERE + HAVING
# ==================================================

q1 = """
SELECT customer_id,
       SUM(order_amount) AS total_spend
FROM orders
WHERE status='completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
"""

df1 = pd.read_sql_query(q1, conn)

print("=" * 50)
print("Q1")
print("Customers:", len(df1))

# ==================================================
# Q2 - Three Table Join
# ==================================================

q2 = """
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status='completed'
GROUP BY c.city,p.category
"""

df2 = pd.read_sql_query(q2, conn)

print("=" * 50)
print("Q2")
print("Rows:", len(df2))
print("Revenue:", round(df2["revenue"].sum(), 2))

# ==================================================
# Q3 - Scalar Subquery
# ==================================================

q3 = """
SELECT *
FROM products
WHERE price_inr >
(
    SELECT AVG(price_inr)
    FROM products
)
"""

df3 = pd.read_sql_query(q3, conn)

print("=" * 50)
print("Q3")
print("Products:", len(df3))

# ==================================================
# Q4 - LEFT JOIN
# ==================================================

q4 = """
SELECT c.customer_id
FROM customers c
LEFT JOIN
(
    SELECT DISTINCT customer_id
    FROM orders
    WHERE status='completed'
) o
ON c.customer_id = o.customer_id
WHERE o.customer_id IS NULL
"""

df4 = pd.read_sql_query(q4, conn)

print("=" * 50)
print("Q4")
print("Customers:", len(df4))

# ==================================================
# Q5 - CTE
# ==================================================

q5 = """
WITH customer_spend AS
(
    SELECT customer_id,
           SUM(order_amount) AS total_spend
    FROM orders
    WHERE status='completed'
    GROUP BY customer_id
),
avg_spend AS
(
    SELECT AVG(total_spend) AS avg_total
    FROM customer_spend
)
SELECT *
FROM customer_spend
WHERE total_spend >
(
    SELECT avg_total
    FROM avg_spend
)
"""

df5 = pd.read_sql_query(q5, conn)

print("=" * 50)
print("Q5")
print("Customers:", len(df5))

# ==================================================
# Q6 - Ranking Window Functions
# ==================================================

q6 = """
WITH customer_spend AS
(
    SELECT
        c.customer_id,
        c.city,
        ROUND(SUM(o.order_amount),2) AS total_spend
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    WHERE o.status='completed'
    GROUP BY c.customer_id, c.city
)

SELECT
    customer_id,
    city,
    total_spend,

    RANK() OVER
    (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS rank_val,

    DENSE_RANK() OVER
    (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_rank_val,

    ROW_NUMBER() OVER
    (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS row_num

FROM customer_spend
"""

df6 = pd.read_sql_query(q6, conn)

print("=" * 50)
print("Q6")

top_customers = df6[df6["row_num"] == 1]

print(
    top_customers[
        ["city", "customer_id", "total_spend"]
    ]
)

rank_diff = (df6["rank_val"] != df6["row_num"]).sum()

print("\nRows where RANK != ROW_NUMBER:")
print(rank_diff)

# ==================================================
# Q7 - Running Total Revenue
# ==================================================

q7 = """
WITH daily_revenue AS
(
    SELECT
        SUBSTR(order_ts,1,10) AS day,
        ROUND(SUM(order_amount),2) AS revenue
    FROM orders
    WHERE status='completed'
    GROUP BY day
)

SELECT
    day,
    revenue,

    SUM(revenue) OVER
    (
        ORDER BY day
    ) AS cumulative_revenue

FROM daily_revenue
ORDER BY day
"""

df7 = pd.read_sql_query(q7, conn)

print("=" * 50)
print("Q7")

print("Number of Days:")
print(len(df7))

print("\nFinal Cumulative Revenue:")
print(round(df7.iloc[-1]["cumulative_revenue"], 2))

# ==================================================
# CLOSE CONNECTION
# ==================================================

conn.close()