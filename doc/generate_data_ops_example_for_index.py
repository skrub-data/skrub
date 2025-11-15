"""
A script to generate pipeline snippets for the skrub index page.

The first and second "cells" of this script are the actual demo, for the index page.

The other cells are used to generate the HTML snippets for the index page.
"""
# We want manual control over the formatting as those snippets are shown in the home page
# fmt: off
# ruff: noqa: I001, E402, E501

# A print statement to show that the script is running
print("Generating pipeline snippets for the skrub index page...")

# %%
# A dataset with multiple tables
import skrub
from skrub.datasets import fetch_credit_fraud
# use the test set to have smaller data
dataset = fetch_credit_fraud(split="test")

# Extract simplified tables
# Drop the columns that are not central to the analysis
products_df = dataset.products[["basket_ID", "cash_price", "Nbr_of_prod_purchas"]]
# Rename the ID column to "basket_ID"
baskets_df = dataset.baskets.rename(columns={"ID": "basket_ID"})

# %%
# Save HTML snippets of the tables
import pathlib
OUTPUT_DIR = pathlib.Path("generated_for_index")
# Create the output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

products_df.to_html(OUTPUT_DIR / "products.html", index=False, max_rows=4)
baskets_df.to_html(OUTPUT_DIR / "baskets.html", index=False, max_rows=4)

# %%
# Define the inputs of our skrub pipeline
products = skrub.var("products", products_df)
baskets = skrub.var("baskets", baskets_df)

# Specify our "X" and "y" variables for machine learning
basket_IDs = baskets[["basket_ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

# A pandas-based data-preparation pipeline that merges the tables
aggregated_products = products.groupby("basket_ID").agg(
    skrub.choose_from(("mean", "max", "count"))).reset_index()
features = basket_IDs.merge(aggregated_products, on="basket_ID")
from sklearn.ensemble import ExtraTreesClassifier
predictions = features.skb.apply(ExtraTreesClassifier(), y=fraud_flags)

# Now use skrub to tune hyperparameters of the above pipeline
search = predictions.skb.make_grid_search(fitted=True, scoring="roc_auc")
search.plot_results()

# %%
# Save the above figure
fig = search.plot_results()
with open(OUTPUT_DIR / "parallel_coordinates.html", "w") as f:
    f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))


# Save a graph of the pipeline
graph = predictions.skb.draw_graph()
with open(OUTPUT_DIR / "pipeline.svg", "wb") as f:
    f.write(graph.svg)


# %%
# Now save html snippets of this notebook

# Retrieve the code of the current file
import os

this_file = os.path.abspath(__file__)
print(this_file)
# Read the file and split into lines
with open(this_file, "r") as f:
    lines = f.readlines()


# Extract the code from the lines
def extract_code(lines):
    code_blocks = []
    this_code_block = []
    in_code_block = False
    for line in lines:
        if line.startswith("# %%"):
            in_code_block = True
            if this_code_block:
                code_blocks.append("".join(this_code_block))
                this_code_block = []
            continue
        if in_code_block:
            this_code_block.append(line)
    return code_blocks


# Extract the code from the lines
code_blocks = extract_code(lines)

# Save syntax-highlighted code snippets as HTML
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

for n in [0, 2]:
    with open(OUTPUT_DIR / f"code_block_{n}.html", "w") as f:
        f.write(highlight(code_blocks[n], PythonLexer(), HtmlFormatter()))


# fmt: on
