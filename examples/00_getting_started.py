"""
Getting Started
===============

This guide showcases the features of ``skrub``, an open-source package that aims at
bridging the gap between tabular data sources and machine-learning models.


Much of ``skrub`` revolves around vectorizing, assembling, and encoding tabular data,
to prepare data in a format that shallow or classic machine-learning models understand.
"""

# %%
# Downloading example datasets
# ----------------------------
#
# The :obj:`~skrub.datasets` module allows us to download tabular datasets and
# demonstrate ``skrub``'s features.

# %%
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
employees_df, salaries = dataset.X, dataset.y

# %%
# Explore all the available datasets in :ref:`downloading_a_dataset_ref`.


# %%
# Generating an interactive report for a dataframe
# -------------------------------------------------
#
# To quickly get an overview of a dataframe's contents, use the
# :class:`~skrub.TableReport`.

# %%
from skrub import TableReport

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
# Easily building a strong baseline for tabular machine learning
# --------------------------------------------------------------
#
# The goal of ``skrub`` is to ease tabular data preparation for machine learning.
# The :func:`~skrub.tabular_learner` function provides an easy way to build a simple
# but reliable machine-learning model, working well on most tabular data.


# %%
from sklearn.model_selection import cross_validate

from skrub import tabular_learner

model = tabular_learner("regressor")
results = cross_validate(model, employees_df, salaries)
results["test_score"]

# %%
# To handle rich tabular data and feed it to a machine-learning model, the
# pipeline returned by :func:`~skrub.tabular_learner` preprocesses and encodes
# strings, categories and dates using the :class:`~skrub.TableVectorizer`.
# See its documentation or :ref:`sphx_glr_auto_examples_01_encodings.py` for
# more details. An overview of the chosen defaults is available in
# :ref:`end_to_end_pipeline`.


# %%
# Assembling data
# ---------------
#
# ``Skrub`` allows imperfect assembly of data, such as joining dataframes
# on columns that contain typos. ``Skrub``'s joiners have ``fit`` and
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
# notice the "Rome" instead of "Roma"
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
# It's also possible to augment data by joining and aggregating multiple
# dataframes with the :class:`~skrub.AggJoiner`. This is particularly useful to
# summarize information scattered across tables, for instance adding statistics
# about flights to the dataframe of airports:

# %%
from skrub import AggJoiner

flights = pd.DataFrame(
    {
        "flight_id": range(1, 7),
        "from_airport": [1, 1, 1, 2, 2, 2],
        "total_passengers": [90, 120, 100, 70, 80, 90],
        "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
    }
)
agg_joiner = AggJoiner(
    aux_table=flights,
    main_key="airport_id",
    aux_key="from_airport",
    cols=["total_passengers", "company"],  # the cols to perform aggregation on
    operations=["mean", "mode"],  # the operations to compute
)
agg_joiner.fit_transform(airports)

# %%
# For joining multiple auxiliary tables on a main table at once, use the
# :class:`~skrub.MultiAggJoiner`.
#
# See other ways to join multiple tables in :ref:`assembling`.


# %%
# Encoding data
# -------------
#
# When a column contains categories with variations and typos, it can
# be encoded using one of ``skrub``'s encoders, such as the
# :class:`~skrub.GapEncoder`.
#
# The :class:`~skrub.GapEncoder` creates a continuous encoding, based on
# the activation of latent categories. It will create the encoding based on
# combinations of substrings which frequently co-occur.
#
# For instance, we might want to encode a column ``X`` that contains
# information about cities, being either Madrid or Rome :

# %%
from skrub import GapEncoder

X = pd.Series(
    [
        "Rome, Italy",
        "Rome",
        "Roma, Italia",
        "Madrid, SP",
        "Madrid, spain",
        "Madrid",
        "Romq",
        "Rome, It",
    ],
    name="city",
)
enc = GapEncoder(n_components=2, random_state=0)  # 2 topics in the data
enc.fit(X)

# %%
# The :class:`~skrub.GapEncoder` has found the following two topics:

# %%
enc.get_feature_names_out()

# %%
# Which correspond to the two cities.
#
# Let's see the activation of each topic depending on the rows of ``X``:

# %%
encoded = enc.fit_transform(X).assign(original=X)
encoded

# %%
# The higher the activation, the closer the row to the latent topic. These
# columns can now be understood by a machine-learning model.
#
# The other encoders are presented in :ref:`encoding`.


# %%
# Next steps
# ----------
#
# We have briefly covered pipeline creation, vectorizing, assembling, and encoding
# data. We presented the main functionalities of ``skrub``, but there is much
# more to it !
#
# Please refer to our :ref:`user_guide` for a more in-depth presentation of
# ``skrub``'s concepts, or visit our
# `examples <https://skrub-data.org/stable/auto_examples>`_ for more
# illustrations of the tools that we provide !
