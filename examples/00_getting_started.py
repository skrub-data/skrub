"""
Getting Started
===============

This guide showcases some of the features of ``skrub``, an open-source package
that aims at bridging the gap between tabular data stored in Pandas or Polars
dataframes, and machine-learning models.

Much of ``skrub`` revolves around simplifying many of the tasks that are involved
in pre-processing raw data into a format that shallow or classic machine-learning
models can understand, that is, numerical data.

``skrub`` does this by vectorizing, assembling, and encoding tabular data through
a number of features that we present in this example and the following.
"""

# %%
# Downloading example datasets
# ----------------------------
#
# The :obj:`~skrub.datasets` module allows us to download tabular datasets and
# demonstrate ``skrub``'s features.
#
# .. note::
#
#    You can control the directory where the datasets are stored by:
#
#    - setting in your environment the ``SKRUB_DATA_DIRECTORY`` variable to an
#      absolute directory path,
#    - using the parameter ``data_directory`` in fetch functions, which takes
#      precedence over the envar.
#
#    By default, the datasets are stored in a folder named "skrub_data" in the
#    user home folder.


# %%
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
employees_df, salaries = dataset.X, dataset.y

# %%
# Explore all the available datasets in :ref:`datasets_ref`.

# %%
# Preliminary exploration and parsing of data
# -------------------------------------------------
# Typically, the first operations that are done on new data involve data exploration
# and parsing.
# To quickly get an overview of a dataframe's contents, use the
# :class:`~skrub.TableReport`.
# Here, we also use the :class:`~skrub.Cleaner`, a transformer that cleans the
# dataframe by parsing nulls and dates, and by dropping "uninformative" columns
# (e.g., that contain too many nulls, or that are constant).
#

# %%
from skrub import Cleaner, TableReport

TableReport(employees_df)

# %%
# From the Report above, we can see that there are datetime columns, so we use the
# :class:`~skrub.Cleaner` to parse them.

employees_df = Cleaner().fit_transform(employees_df)
TableReport(employees_df)

# %%
#
# You can use the interactive display above to explore the dataset visually.
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

# %%
# It is also possible to tell ``skrub`` to replace the default pandas & polars
# displays with ``TableReport`` by modifying the global config with
# :func:`~skrub.set_config`.

from skrub import set_config

set_config(use_table_report=True)

employees_df

# %%
# This setting can easily be reverted:

set_config(use_table_report=False)

employees_df

# %%
# Easily building a strong baseline for tabular machine learning
# --------------------------------------------------------------
#
# The goal of ``skrub`` is to ease tabular data preparation for machine learning.
# The :func:`~skrub.tabular_pipeline` function provides an easy way to build a simple
# but reliable machine learning model that works well on most tabular data.


# %%
from sklearn.model_selection import cross_validate

from skrub import tabular_pipeline

model = tabular_pipeline("regressor")
results = cross_validate(model, employees_df, salaries)
results["test_score"]

# %%
# To handle rich tabular data and feed it to a machine learning model, the
# pipeline returned by :func:`~skrub.tabular_pipeline` preprocesses and encodes
# strings, categories and dates using the :class:`~skrub.TableVectorizer`.
# See its documentation or :ref:`sphx_glr_auto_examples_01_encodings.py` for
# more details. An overview of the chosen defaults is available in
# :ref:`userguide_tablevectorizer`.


# %%
# Assembling data
# ---------------
#
# ``skrub`` allows imperfect assembly of data, such as joining dataframes
# on columns that contain typos. ``skrub``'s joiners have ``fit`` and
# ``transform`` methods, storing information about the data across calls.
#
# The :class:`~skrub.Joiner` allows fuzzy-joining multiple tables, each row of
# a main table will be augmented with values from the best match in the auxiliary table.
# You can control how distant fuzzy-matches are allowed to be with the
# ``max_dist`` parameter.

# %%
# In the following, we add information about countries to a table containing
# airports and the cities they are in:

# %%
import pandas as pd

from skrub import Joiner

airports = pd.DataFrame(
    {
        "airport_id": [1, 2],
        "airport_name": ["Charles de Gaulle", "Aeroporto Leonardo da Vinci"],
        "city": ["Paris", "Roma"],
    }
)
# Notice the "Rome" instead of "Roma"
capitals = pd.DataFrame(
    {"capital": ["Berlin", "Paris", "Rome"], "country": ["Germany", "France", "Italy"]}
)
joiner = Joiner(
    capitals,
    main_key="city",
    aux_key="capital",
    max_dist=0.8,
    add_match_info=False,
)
joiner.fit_transform(airports)

# %%
# Information about countries have been added, even if the rows aren't exactly matching.
#
# ``skrub`` allows to aggregate multiple tables according to various strategies: you
# can see other ways to join multiple tables in :ref:`userguide_joining_tables`.

# %%
# Encoding any data as numerical features
# ---------------------------------------
#
# Tabular data can contain a variety of datatypes, ranging from numerical, to
# datetimes, to categories, strings, and text. Encoding features in a meaningful
# way requires a lot of effort and is a major part of the feature engineering
# process that is required to properly train machine learning models.
#
# ``skrub`` helps with this by providing various transformers that automatically
# encode different datatypes into ``float32`` features.
#
# For **numerical features**, the :class:`~skrub.SquashingScaler` applies a robust
# scaling technique that is less sensitive to outliers. Check the
# :ref:`relative example <sphx_glr_auto_examples_11_squashing_scaler.py>`
# for more information on the feature.
#
# For **datetime columns**, ``skrub`` provides the :class:`~skrub.DatetimeEncoder`
# which can extract useful features such as year, month, day, as well as additional
# features such as weekday or day of year. Periodic encoding with trigonometric
# or spline features is also available. Refer to the :class:`~skrub.DatetimeEncoder`
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
# ``skrub`` transformers are applied column-by-column, but it is possible to use
# the :class:`~skrub.ApplyToCols` meta-transformer to apply a transformer to
# multiple columns at once. Complex column selection is possible using
# :ref:`skrub's column selectors <userguide_selectors>`.

from skrub import ApplyToCols, DatetimeEncoder

ApplyToCols(
    DatetimeEncoder(add_total_seconds=False), cols=["date_1", "date_2"]
).fit_transform(data)

# %%
# Finally, when a column contains **categorical or string data**, it can be
# encoded using various encoders provided by ``skrub``. The default encoder is
# the :class:`~skrub.StringEncoder`, which encodes categories using
# `Latent Semantic Analysis (LSA) <https://scikit-learn.org/stable/modules/decomposition.html#about-truncated-svd-and-latent-semantic-analysis-(lsa)>`_.
# It is a simple and efficient way to encode categories, and works well in
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
# :class:`~skrub.TextEncoder`,
# which uses pre-trained language models retrieved from the HuggingFace hub to
# create meaningful text embeddings.
# See :ref:`userguide_encoders` for more details on all the categorical encoders
# provided by ``skrub``, and :ref:`sphx_glr_auto_examples_01_encodings.py` for a
# comparison between the different methods.

# %%
# Advanced use cases
# ----------------------
# If your use case involves more complex data preparation, hyperparameter tuning,
# or model selection, if you want to build a multi-table pipeline that requires
# assembling and preparing multiple tables, or if you want to make sure that the
# data preparation can be reproduced exactly, you can use the ``skrub`` Data Ops,
# a powerful framework which provides tools to build complex data processing pipelines.
# See the relative :ref:`user guide <userguide_data_ops>` and the
# :ref:`data_ops_examples_ref`
# examples for more details.

# %%
# Next steps
# ----------
#
# We have briefly covered pipeline creation, vectorizing, assembling, and encoding
# data. We presented the main functionalities of ``skrub``, but there is much
# more to it!
#
# Please refer to our :ref:`user_guide` for a more in-depth presentation of
# ``skrub``'s concepts, or visit our
# `examples <https://skrub-data.org/stable/auto_examples>`_ for more
# illustrations of the tools that we provide!
#
