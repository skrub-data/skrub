"""
Hands-On with Column Selection and Transformers
===============================================

In previous examples, we saw how skrub provides powerful abstractions like
:class:`~skrub.TableVectorizer` and :func:`~skrub.tabular_pipeline` to create pipelines.

In this new example, we show how to create more flexible pipelines by selecting
and transforming dataframe columns using arbitrary logic.
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
# Our goal is now to apply a :class:`~skrub.StringEncoder` to two columns of our
# choosing: ``division`` and ``employee_position_title``.
#
# We can achieve this using :class:`~skrub.ApplyToCols`, whose job is to apply a
# transformer to multiple columns independently, and let unmatched columns through
# without changes.
# This can be seen as a handy drop-in replacement of the
# :class:`~sklearn.compose.ColumnTransformer`.
#
# Since we selected two columns and set the number of components to ``30`` each,
# :class:`~skrub.ApplyToCols` will create ``2*30`` embedding columns in the dataframe
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
# In addition to the :class:`~skrub.ApplyToCols` class, the
# :class:`~skrub.ApplyToFrame` class is useful for transformers that work on multiple
# columns at once, such as the :class:`~sklearn.decomposition.PCA` which reduces the
# number of components.
#
# To select columns without hardcoding their names, we introduce
# :ref:`selectors<user_guide_selectors>`, which allow for flexible matching pattern
# and composable logic.
#
# The regex selector below will match all columns prefixed with ``"lsa"``, and pass them
# to :class:`~skrub.ApplyToFrame` which will assemble these columns into a dataframe and
# finally pass it to the PCA.
from sklearn.decomposition import PCA

from skrub import ApplyToFrame
from skrub import selectors as s

apply_pca = ApplyToFrame(PCA(n_components=8), cols=s.regex("lsa"))
Xt = apply_pca.fit_transform(Xt)
Xt

# %%
# These two selectors are scikit-learn transformers and can be chained together within
# a :class:`~sklearn.pipeline.Pipeline`.
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    apply_string_encoder,
    apply_pca,
).fit_transform(X)

# %%
# Note that selectors also come in handy in a pipeline to select or drop columns, using
# :class:`~skrub.SelectCols` and :class:`~skrub.DropCols`!
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
# Suppose we want to apply an :class:`~sklearn.preprocessing.OrdinalEncoder` on
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
# operator ``~`` and apply a :class:`skrub.StringEncoder` to vectorize those
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
# :class:`~skrub.TableVectorizer`, also used in :func:`~skrub.tabular_pipeline`.
#
# Click on the dropdown arrows next to the datatype to see the columns are mapped to
# the different transformers in :class:`~skrub.TableVectorizer`.
from skrub import tabular_pipeline

tabular_pipeline("regressor").fit(X, y)
# %%
