"""
.. _example_feature_augmentation:

===========================================
Chain fuzzy joins with the FeatureAugmenter
===========================================

In this example, we will show how to use the |FeatureAugmenter| to chain
multiple |fuzzy_join| operations over various tables.

We will use the same dataset as in the :ref:`example_fuzzy_joining` example,

.. topic:: Note:

    This example is a continuation of the :ref:`example_fuzzy_joining` example.
    We recommend reading it first.


.. |fuzzy_join| replace::
    :func:`~skrub.fuzzy_join`

.. |FeatureAugmenter| replace::
    :class:`~skrub.FeatureAugmenter`
"""

###############################################################################
# Importing the data
# ------------------
#
# Let's import the same tables as in the previous example:

import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/dirty-cat/datasets/master/data/Happiness_report_2022.csv",
    thousands=",",
)
# This table contains a placeholder row at the end, that we remove:
df = df.drop(df.tail(1).index)
df.head(5)

###############################################################################

from skrub.datasets import fetch_world_bank_indicator

life_expectancy = fetch_world_bank_indicator("SP.DYN.LE00.IN").X
life_expectancy.head(5)

###############################################################################

gdp_per_capita = fetch_world_bank_indicator(indicator_id="NY.GDP.PCAP.CD").X
gdp_per_capita.head(5)

###############################################################################

legal_rights = fetch_world_bank_indicator("IC.LGL.CRED.XQ").X
life_expectancy.head(5)

###############################################################################
# fuzzy joining multiple tables
# -----------------------------
#
# Instead of manually performing the multiple fuzzy_joins like we
# did previously, we can merge them using the |FeatureAugmenter|.
#
# We gather the auxiliary tables into a list of 2-tuples (tables, keys)
# for the `tables` parameter.

from skrub import FeatureAugmenter

fa = FeatureAugmenter(
    tables=[
        (gdp_per_capita, "Country Name"),
        (life_expectancy, "Country Name"),
        (legal_rights, "Country Name"),
    ],
    main_key="Country",
)

###############################################################################
# Notice we haven't passed the main table yet.
# This is by design, as we want this encoder to be scikit-learn compatible.
# We will instead pass it when calling :func:`~skrub.FeatureAugmenter.fit`.
#
# To get our final joined table we will fit and transform the main table
# with our instance of the |FeatureAugmenter|:

X = fa.fit_transform(df)
X.head(10)

###############################################################################
# And that's it! We have the same result as in the previous example,
# where we did it all manually.
#
# We now have a rich table ready for machine learning.
# Let's create our machine learning pipeline
# (the same as in the previous example):

from sklearn import __version__ as sklearn_version

if sklearn_version < "1.0":
    from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

# We include only the columns that will be pertinent for our regression:
column_filter = make_column_transformer(
    (
        "passthrough",
        [
            "GDP per capita (current US$)",
            "Life expectancy at birth, total (years)",
            "Strength of legal rights index (0=weak to 12=strong)",
        ],
    ),
    remainder="drop",
)

pipeline = make_pipeline(
    fa,
    column_filter,
    HistGradientBoostingRegressor(),
)

###############################################################################
# And maybe the best part is that we are now able to evaluate
# the parameters of the |fuzzy_join|.
#
# For instance, the ``match_score`` was previously manually picked and can now
# be tested using a grid search:

from sklearn.model_selection import GridSearchCV

y = df["Happiness score"]

# We will test a few possible values of match_score
params = {"featureaugmenter__match_score": [0.3, 0.4, 0.5, 0.6, 0.7]}

grid = GridSearchCV(pipeline, param_grid=params)
grid.fit(df, y)

print(grid.best_params_)

###############################################################################
# Great, we found the best ``match_score``!
# Let's see how it affects our model's performance:

print(f"Mean R2 score with pipeline is {grid.score(df, y):.2f}")

###############################################################################
#
# .. topic:: Note:
#
#     Here, ``grid.score()`` takes directly the best model
#     (with ``match_score=0.5``) that was found during the grid search.
#     Thus, it is equivalent to fixing the ``match_score`` to 0.5 and
#     refitting the pipeline on the data.
#
#
# Great, by evaluating the correct ``match_score`` we improved our
# results significantly!
