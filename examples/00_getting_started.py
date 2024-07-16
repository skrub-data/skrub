"""
Getting Started
===============

The purpose of this guide is to provide an introduction to the functionalities of
``skrub``, an open-source package that aims at bridging the gap between tabular
data sources and machine-learning models. Please refer to our `installation guidelines
<https://skrub-data.org/stable/install.html>`_ for installing ``skrub``.

Much of ``skrub`` revolves around vectorizing, assembling, and encoding tabular data,
to prepare data in a format that machine-learning models understand.
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
# Explore all the available `datasets
# <https://skrub-data.org/stable/reference/downloading_a_dataset>`_.


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
# You can use the interactive display above to explore the dataset visually.

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
# `end-to-end pipeline <https://skrub-data.org/stable/end_to_end_pipeline>`_.


# %%
# Assembling data
# ---------------
#
# ``Skrub`` allows imperfect assembly of data, in the case where columns are dirty
# and can contain typos.
#
# The :class:`~skrub.Joiner` allows to fuzzy-join multiple tables, and each row of
# a main table will be augmented with values from the best match in the auxiliary table.
# You can control how distant fuzzy-matches are allowed to be with the
# ``max_dist`` parameter.

# %%
# In the following, we add information about capitals to a table of countries:

# %%
import pandas as pd

from skrub import Joiner

main_table = pd.DataFrame({"Country": ["France", "Italia", "Georgia"]})
# notice the "Italy" instead of "Italia"
aux_table = pd.DataFrame(
    {"Country": ["Germany", "France", "Italy"], "Capital": ["Berlin", "Paris", "Rome"]}
)
joiner = Joiner(
    aux_table,
    key="Country",
    suffix="_aux",
    max_dist=0.8,
    add_match_info=False,
)
joiner.fit_transform(main_table)

# %%
# Information about capitals have been added for France and Italia, but not for Georgia
# since it's not present in the auxiliary table.
#
# It's also possible to augment data by joining and aggregating multiple
# dataframes with the :class:`~skrub.AggJoiner`. This is particularly useful to
# summarize information scattered across tables:

# %%
from skrub import AggJoiner

main = pd.DataFrame(
    {
        "airportId": [1, 2],
        "airportName": ["Paris CDG", "NY JFK"],
    }
)
aux = pd.DataFrame(
    {
        "flightId": range(1, 7),
        "from_airport": [1, 1, 1, 2, 2, 2],
        "total_passengers": [90, 120, 100, 70, 80, 90],
        "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
    }
)
agg_joiner = AggJoiner(
    aux_table=aux,
    main_key="airportId",
    aux_key="from_airport",
    cols=["total_passengers", "company"],  # the cols to perform aggregation on
    operations=["mean", "mode"],  # the operations to compute
)
agg_joiner.fit_transform(main)

# %%
# For joining multiple auxiliary tables on a main table at once, use the
# :class:`~skrub.MultiAggJoiner`.
#
# See other ways to join multiple tables on
# `assembling data <https://skrub-data.org/stable/assembling>`_.


# %%
# Encoding data
# -------------
#
# When a column contains dirty categories, it can be encoded using one
# of ``skrub``'s encoders, such as the :class:`~skrub.GapEncoder`.
#
# The :class:`~skrub.GapEncoder` creates a continuous encoding, based on
# the activation of latent categories. It will create the encoding based on
# combinations of substrings which frequently co-occur.
#
# For instance, we might want to encode a column ``X`` that we know contains
# information about cities, being either Madrid or Rome :

# %%
from skrub import GapEncoder

enc = GapEncoder(n_components=2, random_state=0)  # 2 topics in the data

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
enc.fit(X)

# %%
# The :class:`~skrub.GapEncoder` has found the following two topics:

# %%
enc.get_feature_names_out()

# %%
# Which correspond to the two cities.
#
# Let's see the activation of each topic in each of the rows of ``X``:

# %%
out = enc.transform(X)
out

# %%
# The higher the activation, the closer the row to the latent topic. These
# activations can then be used to encode ``X``, for instance with a 0 if the
# city is Madrid, and 1 if the city is Rome:

# %%
madrid = out.iloc[:, 0] > out.iloc[:, 1]
X[madrid] = 0
X[~madrid] = 1
X

# %%
# Which correspond to the respective positions of Madrid and Rome in the initial
# column ! This column can now be understood by a machine-learning model.
#
# The other encoders are presented in `encoding <https://skrub-data.org/stable/encoding>`_.


# %%
# Next steps
# ----------
#
# We have briefly covered pipeline creation, vectorizing, assembling, and encoding
# data. We presented the main functionalities of ``skrub``, but there is much
# more to it !
#
# Please refer to our `User Guide <https://skrub-data.org/stable/documentation>`_
# for a more in-depth presentation of ``skrub``'s concepts. You can also check out
# our `API reference <https://skrub-data.org/stable/api>`_ for the exhaustive list
# of functionalities !
#
# Visit our `examples <https://skrub-data.org/stable/auto_examples>`_ for more
# illustrations of the tools offered by ``skrub``.
