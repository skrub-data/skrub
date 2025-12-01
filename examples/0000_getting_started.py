"""
Getting Started
===============

This guide showcases some of the features of skrub.
Much of skrub revolves around simplifying many of the tasks that are involved
in pre-processing raw data into a format that shallow or classic machine-learning
models can understand, that is, numerical data.

Skrub achieves this by vectorizing, assembling, and encoding tabular data through
the features we present in this example and the following ones.

.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |set_config| replace:: :func:`~skrub.set_config`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |Joiner| replace:: :class:`~skrub.Joiner`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |DatetimeEncoder| replace:: :class:`~skrub.DatetimeEncoder`
.. |ApplyToCols| replace:: :class:`~skrub.ApplyToCols`
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
"""

# %%
# Preliminary exploration with the |TableReport|
# ----------------------------------------------
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
employees_df, salaries = dataset.X, dataset.y

# %%
# Typically, the first step with new data is exploration and parsing.
# To quickly get an overview of a dataframe's contents, use the |TableReport|.

# %%
from skrub import TableReport

TableReport(employees_df)

# %%
# You can use the interactive display above to explore the dataset visually.
#
# It is also possible to tell skrub to replace the default pandas and polars
# displays with |TableReport| by modifying the global config with
# |set_config|.
#
# .. note::
#
#    You can see a few more `example reports`_ online. We also
#    provide an experimental online demo_ that allows you to select a CSV or
#    parquet file and generate a report directly in your web browser, without
#    installing anything.
#
#    .. _example reports: https://skrub-data.org/skrub-reports/examples/
#    .. _demo: https://skrub-data.org/skrub-reports/
#
# From the report above, we see that there are columns with date and time stored
# as `object` dtype (cf. "Stats" tab of the report).
# Datatypes not being parsed correctly is a scenario that occurs commonly after
# reading a table. We can use the |Cleaner| to address this.
# In the next section, we show that this transformer does additional cleaning.

# %%
# Sanitizing data with the |Cleaner|
# ----------------------------------
# Here, we use the |Cleaner|, a transformer that sanitizing the
# dataframe by parsing nulls and dates, and by dropping "uninformative" columns
# (e.g., columns with too many nulls or that are constant).
#

from skrub import Cleaner

employees_df = Cleaner().fit_transform(employees_df)
TableReport(employees_df)

# %%
# We can see from the "Stats" tab that now the column `date_first_hired` has been
# parsed correctly as a Datetime.

# %%
# Easily building a strong baseline for tabular machine learning
# --------------------------------------------------------------
#
# The goal of skrub is to ease tabular data preparation for machine learning.
# The |tabular_pipeline| function provides an easy way to build a simple
# but reliable machine learning model that works well on most tabular data.


# %%
from sklearn.model_selection import cross_validate

from skrub import tabular_pipeline

model = tabular_pipeline("regressor")
model
# %%
results = cross_validate(model, employees_df, salaries)
results["test_score"]

# %%
# To handle rich tabular data and feed it to a machine learning model, the
# pipeline returned by |tabular_pipeline| preprocesses and encodes
# strings, categories and dates using the |TableVectorizer|.
# See its documentation or :ref:`sphx_glr_auto_examples_0010_encodings.py` for
# more details. An overview of the chosen defaults is available in
# :ref:`user_guide_tabular_pipeline`.


# %%
# Encoding any data as numerical features
# ---------------------------------------
#
# Tabular data can contain a variety of datatypes, from numerical to
# datetimes, categories, strings, and text. Encoding features in a meaningful
# way requires significant effort and is a major part of the feature engineering
# process required to properly train machine learning models.
#
# Skrub helps with this by providing various transformers that automatically
# encode different datatypes into ``float32`` features.
#
# For **numerical features**, the |SquashingScaler| applies a robust
# scaling technique that is less sensitive to outliers. Check the
# :ref:`relative example <sphx_glr_auto_examples_0100_squashing_scaler.py>`
# for more information on the feature.
#
# For **datetime columns**, skrub provides the |DatetimeEncoder|
# which can extract useful features such as year, month, day, as well as additional
# features such as weekday or day of year. Periodic encoding with trigonometric
# or spline features is also available. Refer to the |DatetimeEncoder|
# documentation for more detail.
#

# %%
import pandas as pd

data = pd.DataFrame(
    {
        "event": ["A", "B", "C"],
        "date_1": ["2020-01-01", "2020-06-15", "2021-03-22"],
        "date_2": ["2020-01-15", "2020-07-01", "2021-04-05"],
    }
)
data = Cleaner().fit_transform(data)
TableReport(data)
# %%
# Skrub transformers are applied column-by-column, but it's possible to use
# the |ApplyToCols| meta-transformer to apply a transformer to
# multiple columns at once. Complex column selection is possible using
# :ref:`skrub's column selectors <user_guide_selectors>`.

from skrub import ApplyToCols, DatetimeEncoder

ApplyToCols(
    DatetimeEncoder(add_total_seconds=False), cols=["date_1", "date_2"]
).fit_transform(data)

# %%
# Finally, when a column contains **categorical or string data**, it can be
# encoded using various encoders provided by skrub. The default encoder is
# the |StringEncoder|, which encodes categories using
# `Latent Semantic Analysis (LSA) <https://scikit-learn.org/stable/modules/decomposition.html#about-truncated-svd-and-latent-semantic-analysis-(lsa)>`_.
# It is a simple and efficient way to encode categories and works well in
# practice.

data = pd.DataFrame(
    {
        "city": ["Paris", "London", "Berlin", "Madrid", "Rome"],
        "country": ["France", "UK", "Germany", "Spain", "Italy"],
    }
)
TableReport(data)
from skrub import StringEncoder

StringEncoder(n_components=3).fit_transform(data["city"])

# %%
# If your data includes a lot of text, you may want to use the
# |TextEncoder|,
# which uses pre-trained language models retrieved from the HuggingFace hub to
# create meaningful text embeddings.
# See :ref:`user_guide_encoders_index` for more details on all the categorical encoders
# provided by skrub, and :ref:`sphx_glr_auto_examples_0010_encodings.py` for a
# comparison between the different methods.
#

# %%
# Assembling data
# ---------------
#
# Skrub allows imperfect assembly of data, such as joining dataframes
# on columns that contain typos. Skrub's joiners have ``fit`` and
# ``transform`` methods, storing information about the data across calls.
#
# The |Joiner| allows fuzzy-joining multiple tables, where each row of
# a main table will be augmented with values from the best match in the auxiliary table.
# You can control how distant fuzzy-matches are allowed to be with the
# ``max_dist`` parameter.
#
# Skrub also allows you to aggregate multiple tables according to various strategies.
# You can see other ways to join multiple tables in
# :ref:`user_guide_joining_dataframes`.

# %%
# Advanced use cases
# ----------------------
# If your use case involves more complex data preparation, hyperparameter tuning,
# or model selection, if you want to build a multi-table pipeline that requires
# assembling and preparing multiple tables, or if you want to ensure that the
# data preparation can be reproduced exactly, you can use the skrub Data Ops,
# a powerful framework that provides tools to build complex data processing pipelines.
# See the related :ref:`user guide <user_guide_data_ops_index>` and the
# :ref:`data_ops_examples_ref`
# examples for more details.

# %%
# Next steps
# ----------
#
# We have briefly covered pipeline creation, vectorizing, assembling, and encoding
# data. We presented the main functionalities of skrub, but there is much
# more to explore!
#
# Please refer to our :ref:`user_guide` for a more in-depth presentation of
# skrub's concepts, or visit our
# `examples <https://skrub-data.org/stable/auto_examples>`_ for more
# illustrations of the tools that we provide!
#
