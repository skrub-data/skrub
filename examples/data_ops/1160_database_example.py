"""
Working with databases using skrub: example SQLite and Ibis integration
=======================================================================
"""

# %%
# This example focuses on integrating database queries into a skrub DataOps plan.
# We do not cover model evaluation or deployment.
#
# In many production settings, tabular data lives in SQL databases rather than
# in-memory pandas DataFrames. skrub is designed to integrate naturally with
# expression-based systems such as Ibis, allowing database-side computation
# before materializing data for machine learning pipelines.
#
# In this example, we demonstrate how to:
#   - connect to a SQLite database using Ibis
#   - perform joins and filtering at the database level
#   - materialize the result once into pandas
#   - build a skrub DataOps plan on top of the database-backed data

# %%
# Setting up a SQLite database
# -----------------------------
# For demonstration purposes, we create a small SQLite database locally.
# In practice, this would typically be an existing production database.

import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd

# Create a temporary SQLite database
db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
db_path = db_file.name
db_file.close()

# Generate example data
np.random.seed(42)
n_customers = 100
n_orders = 500

customers_df = pd.DataFrame(
    {
        "customer_id": range(1, n_customers + 1),
        "age": np.random.randint(18, 80, n_customers),
        "city": np.random.choice(["New York", "London", "Paris", "Tokyo"], n_customers),
    }
)

orders_df = pd.DataFrame(
    {
        "order_id": range(1, n_orders + 1),
        "customer_id": np.random.randint(1, n_customers + 1, n_orders),
        "product": np.random.choice(
            ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"], n_orders
        ),
        "quantity": np.random.randint(1, 5, n_orders),
        "price": np.random.uniform(100, 2000, n_orders).round(2),
    }
)

# Write tables to SQLite
conn = sqlite3.connect(db_path)
customers_df.to_sql("customers", conn, index=False, if_exists="replace")
orders_df.to_sql("orders", conn, index=False, if_exists="replace")
conn.close()

# %%
# Connecting to the database with Ibis
# -----------------------------------
# We use Ibis to define database-side expressions instead of writing raw SQL.

import ibis

con = ibis.sqlite.connect(db_path)

customers = con.table("customers")
orders = con.table("orders")

# %%
# Database-side transformations
# -----------------------------
# In production workflows, joins and filtering are typically executed in the
# database to avoid unnecessary data transfer.

# Join orders with customers
joined = orders.join(
    customers,
    orders.customer_id == customers.customer_id,
)

# Select only the columns needed for downstream processing
joined = joined.select(
    orders.price,
    orders.quantity,
    orders.product,
    customers.age,
    customers.city,
)

# Optionally, apply database-side filtering
joined = joined.filter(joined.price > 200)

# %%
# Materializing the data
# ---------------------
# Once the relevant transformations have been expressed in Ibis, we
# materialize the result as a pandas DataFrame.

joined_df = joined.execute()

# %%
# Building a skrub DataOps plan
# -----------------------------
# We now construct a DataOps plan using the materialized data.

import skrub

X = skrub.X(joined_df)
y = skrub.y(joined_df["price"])

# %%
# Applying a tabular pipeline
# ---------------------------
# For this example, we use skrub's built-in tabular pipeline.
# The goal here is to illustrate integration with database-backed data,
# not to optimize model performance.

tab_pipeline = skrub.tabular_pipeline("regression")

predictions = X.skb.apply(tab_pipeline, y=y)

# %%
# Inspecting the DataOps graph
# ----------------------------
# We can visualize the full DataOps plan to understand how data flows
# from the database materialization to model predictions.

predictions.skb.draw_graph()

# %%
# Conclusion
# ----------
#
# This example demonstrates how skrub can be used alongside database
# backends by:
#
# 1. Defining joins and filters using Ibis expressions
# 2. Materializing the transformed data only once
# 3. Building a skrub DataOps plan on top of the resulting DataFrame
#
# This workflow allows teams to leverage database engines for data
# preparation while keeping a single, reusable machine learning
# pipeline defined with skrub.

# %%
# Cleanup
# -------
# Remove the temporary database file.

os.unlink(db_path)
