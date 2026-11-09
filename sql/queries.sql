SELECT customer_id,
			 SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
	AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

SELECT *
FROM products
WHERE price_inr >
(
 SELECT AVG(price_inr)
 FROM products
);

SELECT c.customer_id
FROM customers c
LEFT JOIN
(
 SELECT DISTINCT customer_id
 FROM orders
 WHERE status='completed'
) o
ON c.customer_id=o.customer_id
WHERE o.customer_id IS NULL;

WITH customer_spend AS
(
    SELECT customer_id,
           SUM(order_amount) total_spend
    FROM orders
    WHERE status='completed'
    GROUP BY customer_id
),
avg_spend AS
(
    SELECT AVG(total_spend) avg_total
    FROM customer_spend
)
SELECT *
FROM customer_spend
WHERE total_spend >
(
    SELECT avg_total
    FROM avg_spend
);

import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")

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

print("Q1:")
print("Customers =", len(df1))

q2 = """
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o
JOIN customers c
ON o.customer_id=c.customer_id
JOIN products p
ON o.product_id=p.product_id
WHERE o.status='completed'
GROUP BY c.city,p.category
ORDER BY revenue DESC
"""

df2 = pd.read_sql_query(q2, conn)

print("\nQ2")
print("Rows =", len(df2))
print("Total Revenue =", round(df2["revenue"].sum(),2))

print(df2.head())

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

print("\nQ3")
print("Products =", len(df3))

q4 = """
SELECT c.customer_id
FROM customers c
LEFT JOIN
(
    SELECT DISTINCT customer_id
    FROM orders
    WHERE status='completed'
) o
ON c.customer_id=o.customer_id
WHERE o.customer_id IS NULL
"""

df4 = pd.read_sql_query(q4, conn)

print("\nQ4")
print("Customers =", len(df4))

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

print("\nQ5")
print("Customers =", len(df5))