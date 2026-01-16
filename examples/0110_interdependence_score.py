"""
interdependence_score : A dependence measure between variables.
===================================================================

The following example illustrates the use of ``:func:`~skrub.interdependence_score```,
a function that measures linear and various nonlinear dependencies between variables.

Its basic idea is to approximate the HSIC (Hilbert Schmidt Independence Criterion)
by computing the first k terms of an infinite dimensional feature map
for the universal gaussian kernel.

We first generate some synthetic variables and then show how the interdependence score
captures the different relationships between them.

"""

# %%
# Generating linear and non linear variables
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skrub import interdependence_score

np.random.seed(42)
n = 1000

v0 = np.random.randn(n)
v1 = 0.8 * v0 + 0.2 * np.random.randn(n)
v2 = np.random.randn(n)
v3 = np.sin(v2)
v4 = np.square(v2)
v5 = np.random.randn(n)
v6 = np.tanh(v0 * v2)
v7 = pd.qcut(
    v1, q=5, labels=["cat1_0", "cat1_1", "cat1_2", "cat1_3", "cat1_4"]
).to_numpy()
v8 = pd.qcut(np.random.randn(n), q=3, labels=["cat2_0", "cat2_1", "cat2_2"]).to_numpy()

v_num = np.column_stack([v0, v1, v2, v3, v4, v5, v6])
v_cat = np.column_stack((v7, v8))
X_dict = {f"v{i}": v_num[:, i] for i in range(v_num.shape[1])}
X_dict.update({f"v{i + 7}": v_cat[:, i] for i in range(v_cat.shape[1])})
X = pd.DataFrame(X_dict)
# %%
# +----------+-------------+---------------+
# | Variable | Type        | Relation      |
# +==========+=============+===============+
# | v0       | random      | v1            |
# +----------+-------------+---------------+
# | v1       | Linear      | v0            |
# +----------+-------------+---------------+
# | v2       | random      | v3 & v4       |
# +----------+-------------+---------------+
# | v3       | Non-linear  | sin(v2)       |
# +----------+-------------+---------------+
# | v4       | Non-linear  | square(v2)    |
# +----------+-------------+---------------+
# | v5       | random      | independent   |
# +----------+-------------+---------------+
# | v6       | Non-linear  | tanh(v0 * v2) |
# +----------+-------------+---------------+
# | v7       | categorical | v1            |
# +----------+-------------+---------------+
# | v8       | categorical | independent   |
# +----------+-------------+---------------+
#
# We then compute the interdependence score.

ids_table, ids_mat = interdependence_score(X, p_val=True, return_matrix=True)
ids_table.head()


# %%
# Analyzing the dependencies measurement
# --------------------------------------------------------------
# To have a better view of the ids scores we look at the matrix.
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(ids_mat)
ax.set_xticks(range(len(ids_mat)), labels=ids_mat.columns)
ax.set_yticks(range(len(ids_mat)), labels=ids_mat.columns)
for i in range(len(ids_mat)):
    for j in range(len(ids_mat)):
        text = ax.text(
            j, i, f"{ids_mat.iloc[i, j]:.2f}", ha="center", va="center", color="w"
        )
plt.title("Interdependence score matrix")
plt.tight_layout()
plt.show()
# %%
# First of all each variable have perfect dependence with itself (ids = 1).
# The linearly dependent variables (v0, v1) have ids greater than 0.90,
# and the non-linear variables as well (v2, v3, v4).
# The variable v6  which is tanh(v0*v2), has a high score with v0 and v2,
# but also with the variables related to them v1, v3, v4.
# The ids also captures the relation between the categorical variable v7 and v1,
# and by extension with v0 because v1 depends on v0.
# Both independent variables v5 and v8 have ids less than 0.1 with all other variables.

# %%
