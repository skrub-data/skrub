"""
.. _example_grid_search_with_the_tablevectorizer:

=================================================
Performing a grid-search with the TableVectorizer
=================================================

In this example, we will see how to customize and grid-search with the
|TableVectorizer|.


.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`

.. |ColumnTransformer| replace:: :class:`~sklearn.compose.ColumnTransformer`

.. |ColumnTransformer| replace:: :class:`~sklearn.compose.make_column_transformer`

"""

###############################################################################
# Customizing the TableVectorizer
# -------------------------------
#
# In this section, we will see two cases where we might want to customize the
# |TableVectorizer|: when we want to use a custom transformer for a column type
# and when we want to use a custom transformer for a specific column.
#
# The data
# ........
#
# Throughout this example, we will use the employee salaries dataset.

from skrub.datasets import fetch_employee_salaries

ds = fetch_employee_salaries()
X = ds.X
y = ds.y

X.head(10)

###############################################################################
# Let's import the |TableVectorizer| and see what the default assignation is:

from skrub import TableVectorizer
from pprint import pprint

tv = TableVectorizer()
tv.fit(X)

pprint(tv.transformers_)

###############################################################################
# Using a custom Transformer for a column type
# ............................................
#
# Say we wanted to use a |OrdinalEncoder| instead of the default for the low
# cardinality categorical columns.
# It is easy to do that by using the dedicated parameter:

from sklearn.preprocessing import OrdinalEncoder

tv = TableVectorizer(
    low_card_cat_transformer=OrdinalEncoder(),
)
tv.fit(X)

###############################################################################
# Have a look at the |TableVectorizer|'s doc for all the available parameters!
#
# Using a custom Transformer for a specific column
# ................................................
#
# Say we wanted to use a |MinHashEncoder| instead of the default for the column
# ``PRODUCTTYPENAME``. Currently, the best way to apply a specific transformer
# to a specific column is to use the |ColumnTransformer|
# (through, for example, |make_column_transformer|):

from skrub import MinHashEncoder
from sklearn.compose import make_column_transformer

tv_mix = make_column_transformer(
    (
        MinHashEncoder(),
        ["PRODUCTTYPENAME"],
    ),
    remainder=TableVectorizer(),
)

###############################################################################
# Notice that we use the |TableVectorizer| as the ``remainder``:
# this means all remaining columns will be passed to the |ColumnTransformer|,
# and encoded automatically!

tv_mix.fit(X)

pprint(tv_mix.transformers_)

###############################################################################
# Grid-searching with the TableVectorizer
# ---------------------------------------
#
# Grid-searching the encoders' hyperparameters contained in the
# |TableVectorizer| is easy!
# For that, we use the dunder separator, which indicates a nesting layer.
# That means that for tuning the parameter ``n_components`` of the
# `GapEncoder` saved in the |TableVectorizer| attribute
# ``high_card_cat_transformer``, we use the syntax
# ``tablevectorizer__high_card_cat_transformer__n_components``.
#
# Here's an example:

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import __version__ as sklearn_version

if sklearn_version < "1.0":
    from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from skrub import GapEncoder

pipeline = make_pipeline(
    TableVectorizer(
        numerical_transformer=StandardScaler(),
        high_card_cat_transformer=GapEncoder(),
    ),
    HistGradientBoostingClassifier(),
)

params = {
    "tablevectorizer__high_card_cat_transformer__n_components": [10, 30, 50],
}

grid_search = GridSearchCV(pipeline, param_grid=params)

###############################################################################
# Conclusion
# ----------
#
# In this notebook, we saw how to better customize the |TableVectorizer| so
# it fits all your needs!
#
# If you've got any improvement ideas, please open a feature request on
# `GitHub <https://github.com/skrub-data/skrub/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml>`_!
# We are always happy to see new suggestions from the community :)
