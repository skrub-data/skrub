"""
Hands-On with Column Selection and Transformers
===============================================

In previous examples, we saw how skrub provides powerful abstractions like
:class:`~skrub.TableVectorizer` and :func:`~skrub.tabular_pipeline` to create pipelines.

In this new example, we show how to create more flexible pipelines by selecting
and transforming dataframe columns using arbitrary logic.

.. |ApplyToCols| replace:: :class:`~skrub.ApplyToCols`
.. |ApplyToEachCol| replace:: :class:`~skrub.ApplyToEachCol`
.. |ApplyToSubFrame| replace:: :class:`~skrub.ApplyToSubFrame`
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |SelectCols| replace:: :class:`~skrub.SelectCols`
.. |DropCols| replace:: :class:`~skrub.DropCols`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`
.. |PCA| replace:: :class:`~sklearn.decomposition.PCA`
.. |Pipeline| replace:: :class:`~sklearn.pipeline.Pipeline`
.. |ColumnTransformer| replace:: :class:`~sklearn.compose.ColumnTransformer`

"""

# %%
# We begin with loading a dataset with heterogeneous datatypes, and replacing Pandas's
# display with the TableReport display via :func:`skrub.set_config`.
import skrub
from skrub.datasets import fetch_employee_salaries

skrub.set_config(use_table_report=True)
data = fetch_employee_salaries()
X, y = data.X, data.y
X

# %%
# Our goal is now to apply a |StringEncoder| to two columns of our
# choosing: ``division`` and ``employee_position_title``.
#
# We can achieve this using |ApplyToCols|, whose job is to apply a
# transformer to multiple columns independently, and let unmatched columns through
# without changes.
# This can be seen as a handy drop-in replacement of the
# |ColumnTransformer|.
#
# Since we selected two columns and set the number of components to ``30`` each,
# |ApplyToCols| will create ``2*30`` embedding columns in the dataframe
# ``Xt``, which we prefix with ``lsa_``.
from skrub import ApplyToCols, StringEncoder

apply_string_encoder = ApplyToCols(
    StringEncoder(n_components=30),
    cols=["division", "employee_position_title"],
    rename_columns="lsa_{}",
)
Xt = apply_string_encoder.fit_transform(X)
Xt

# %%
# The |ApplyToCols| class can detect automatically whether the transformer is a
# ``SingleColumnTransformer`` (i.e., it can only be applied to one column at a time)
# or not, and apply it accordingly. The |StringEncoder| is a ``SingleColumnTransformer``
# and thus applied to each column independently.

# %%
# The |ApplyToCols| class can also be used with transformers that
# can be applied to multiple columns at once, such as the |PCA|.
# Here, we want to use PCA to reduce the number of dimensions of the new ``lsa_``
# columns.
#
# To select columns without hardcoding their names, we introduce
# :ref:`selectors<user_guide_selectors>`, which allow for flexible matching pattern
# and composable logic.
#
# The regex selector below will match all columns prefixed with ``"lsa"``, and pass them
# to |ApplyToCols| which will assemble these columns into a dataframe
# and finally pass it to the PCA
#
# Note that |ApplyToCols| will automatically detect that PCA is not a
# ``SingleColumnTransformer``
# and apply it to the whole sub-dataframe of columns chosen by the selector at once.

from sklearn.decomposition import PCA

from skrub import selectors as s

apply_pca = ApplyToCols(PCA(n_components=8), cols=s.regex("lsa"))
Xt = apply_pca.fit_transform(Xt)
Xt

# %%
# These two selectors are scikit-learn transformers and can be chained together within
# a |Pipeline|.
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    apply_string_encoder,
    apply_pca,
).fit_transform(X)

# %%
# .. dropdown:: Under the hood of |ApplyToCols|
# |ApplyToCols| is implemented using the |ApplyToEachCol| and |ApplyToSubFrame|
# classes.
# The former applies a transformer to each column independently, while the latter
# applies a transformer to a sub-dataframe.
# Normally, users don't need to worry about these two classes, but they can be useful
# when more control is needed.

# %%
# Note that selectors also come in handy in a pipeline to select or drop columns, using
# |SelectCols| and |DropCols|!
from sklearn.preprocessing import StandardScaler

from skrub import SelectCols

# Select only numerical columns
pipeline = make_pipeline(
    SelectCols(cols=s.numeric()),
    StandardScaler(),
).set_output(transform="pandas")
pipeline.fit_transform(Xt)

# %%
# Let's run through one more example to showcase the expressiveness of the selectors.
# Suppose we want to apply an |OrdinalEncoder| on
# categorical columns with low cardinality (e.g., fewer than ``40`` unique values).
#
# We define a column filter using skrub selectors with a lambda function. Note that
# the same effect can be obtained directly by using
# :func:`~skrub.selectors.cardinality_below`.
from sklearn.preprocessing import OrdinalEncoder

low_cardinality = s.filter(lambda col: col.nunique() < 40)
ApplyToCols(OrdinalEncoder(), cols=s.string() & low_cardinality).fit_transform(X)

# %%
# Notice how we composed the selector with :func:`~skrub.selectors.string()`
# using a logical operator. This resulting selector matches string
# columns with cardinality below ``40``.
#
# We can also define the opposite selector ``high_cardinality`` using the negation
# operator ``~`` and apply a |StringEncoder| to vectorize those
# columns.
from sklearn.ensemble import HistGradientBoostingRegressor

high_cardinality = ~low_cardinality
pipeline = make_pipeline(
    ApplyToCols(
        OrdinalEncoder(),
        cols=s.string() & low_cardinality,
    ),
    ApplyToCols(
        StringEncoder(),
        cols=s.string() & high_cardinality,
    ),
    HistGradientBoostingRegressor(),
).fit(X, y)
pipeline

# %%
# Interestingly, the pipeline above is similar to the datatype dispatching performed by
# |TableVectorizer|, also used in :func:`~skrub.tabular_pipeline`.
#
# Click on the dropdown arrows next to the datatype to see the columns are mapped to
# the different transformers in |TableVectorizer|.
from skrub import tabular_pipeline

tabular_pipeline("regressor").fit(X, y)
# %%
