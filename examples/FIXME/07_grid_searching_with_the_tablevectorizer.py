"""
.. _example_grid_search_with_the_tablevectorizer:

=================================================
Performing a grid-search with the TableVectorizer
=================================================

In this example, we will see how to customize the |TableVectorizer|,
and see how we can perform a grid-search with it.


.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`

.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`

.. |GapEncoder| replace:: :class:`~skrub.GapEncoder`

.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`

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

dataset = fetch_employee_salaries()
X = dataset.X
y = dataset.y

X.head(10)

###############################################################################
# Let's import the |TableVectorizer| and see what the default assignation is:

from pprint import pprint

from skrub import TableVectorizer

tv = TableVectorizer()
tv.fit(X)

pprint(tv.transformers_)

###############################################################################
# Using a custom Transformer for a column type
# ............................................
#
# Say we wanted to use a |MinHashEncoder| instead of the default
# |GapEncoder| for the high cardinality categorical columns.
# It is easy to do that by using the dedicated parameter:

from skrub import MinHashEncoder

tv = TableVectorizer(high_cardinality=MinHashEncoder())
tv.fit(X)

pprint(tv.transformers_)

###############################################################################
# If we want to modify what we classify as a high cardinality categorical
# column, we can tweak the ``cardinality_threshold`` parameter.
# Check out the |TableVectorizer|'s doc for more information.
#
# Also have a look at the other types of columns supported by default!
#
# Using a custom Transformer for a specific column
# ................................................
#
# Say we wanted to use a |MinHashEncoder| instead of the default |GapEncoder|,
# but only for the column ``department_name``.
# We can apply a column-specific transformer by using the ``specific_transformers``
# parameter.

tv = TableVectorizer(specific_transformers=[(MinHashEncoder(), ["department_name"])])
tv.fit(X)

pprint(tv.transformers_)

###############################################################################
# Here, for simplicity, we used the unnamed 2-tuple syntax.
#
# You can also give a name to the assignment, as we will see in the next
# section.
#
# Grid-searching with the TableVectorizer
# ---------------------------------------
#
# Grid-searching the encoders' hyperparameters contained in the
# |TableVectorizer| is easy!
# For that, we use the dunder separator, which indicates a nesting layer.
# That means that for tuning the parameter ``n_components`` of the
# |GapEncoder| saved in the |TableVectorizer| attribute
# ``high_cardinality_transformer``, we use the syntax
# ``tablevectorizer__high_cardinality_transformer__n_components``.
#
# We recommend using the 3-tuple syntax for the column-specific transformers,
# which allows us to give a name to the assignment (here ``mh_dep_name``).

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from skrub import GapEncoder

pipeline = make_pipeline(
    TableVectorizer(
        high_cardinality=GapEncoder(),
        specific_transformers=[
            ("mh_dep_name", MinHashEncoder(), ["department_name"]),
        ],
    ),
    HistGradientBoostingClassifier(),
)

params = {
    "tablevectorizer__high_cardinality_transformer__n_components": [10, 30, 50],
    "tablevectorizer__mh_dep_name__n_components": [25, 50],
}

grid_search = GridSearchCV(pipeline, param_grid=params)

###############################################################################
# Conclusion
# ----------
#
# In this notebook, we saw how to better customize the |TableVectorizer| so
# it fits all your needs!
