"""
Quick start to skrub
====================

Skrub provides some tools to easily start analyzing and learning from tabular
data.
"""

# %%
# Downloading example datasets
# ----------------------------
#
# The :obj:`~skrub.datasets` module allows us to download a variety of tabular datasets.

# %%
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
employees, salaries = dataset.X, dataset.y

# %%
# Generating an interactive report for a dataframe
# -------------------------------------------------
#
# To quickly get an overview of a dataframe's contents, use :class:`~skrub.TableReport`.

# %%
from skrub import TableReport

TableReport(employees)

# %%
# You can use the interactive display above to explore the dataset visually.
#
# It is also possible to generate reports from the command-line using
# ``skrub-report ./my_file.csv``. See ``skrub-report --help`` for details.

# %%
# Easily building a strong baseline for tabular machine learning
# --------------------------------------------------------------
#
# The :func:`~skrub.tabular_learner` function provides an easy way to build a
# simple but reliable machine-learning model, working well on most tabular
# data.


# %%
from sklearn.model_selection import cross_validate

from skrub import tabular_learner

model = tabular_learner("regressor")
results = cross_validate(model, employees, salaries)
results["test_score"]

# %%
# To handle rich tabular data and feed it to a machine-learning model, the
# pipeline returned by :func:`~skrub.tabular_learner` preprocesses and encodes
# strings, categories and dates using the :class:`~skrub.TableVectorizer`.
# See its documentation or :ref:`sphx_glr_auto_examples_01_encodings.py` for
# more details.
