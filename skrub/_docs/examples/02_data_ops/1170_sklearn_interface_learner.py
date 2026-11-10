"""
.. currentmodule:: skrub
.. _example_with_sklearn_interface:

Turn a DataOp into a "standard" scikit-learn estimator
======================================================

When our DataOp only needs 2 inputs, X and y, we can get a learner that accepts
the usual scikit-learn inputs rather than a dictionary like SkrubLearners
normally do, by using :meth:`SkrubLearner.with_sklearn_interface` to obtain the
learner wrapped in a :class:`XySkrubLearner`.

"""

# %%
import skrub
from skrub import selectors as s
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

dataset = skrub.datasets.fetch_employee_salaries()

# %%
# Creating an example DataOp
# --------------------------
#
# We define a DataOp that does transformations similar to the
# :class:`TableVectorizer`.

# %%
X = skrub.var("X").skb.apply(skrub.Cleaner())
num = X.skb.select(s.numeric()).skb.set_name("num").skb.apply("passthrough")
low_card = (
    X.skb.select(s.string() & s.cardinality_below(30))
    .skb.set_name("low_card")
    .skb.apply(OrdinalEncoder())
)
high_card = (
    X.skb.select(s.string() - s.cardinality_below(30))
    .skb.set_name("high_card")
    .skb.apply(
        skrub.StringEncoder(
            n_components=skrub.choose_int(10, 20, n_steps=3, name="n_components")
        )
    )
)
datetime = (
    X.skb.select(s.any_date()).skb.set_name("datetime").skb.apply(skrub.DatetimeEncoder())
)
feat = num.skb.concat([low_card, high_card, datetime], axis=1)
feat

# %%
# Creating & using the scikit-learn estimator
# -------------------------------------------
#
# Now we make a learner and ask for the sklearn-like interface. It is used as
# ``tv.fit_transform(X)`` (whereas the default ``.skb.make_learner()`` behavior
# would require ``tv.fit_transform({'X': X})``). On any skrub learner calling
# :meth:`SkrubLearner.with_sklearn_interface` returns such an estimator that
# executes the same DataOp but exposes the scikit-learn (X, y) interface
# instead of the skrub one (which accepts a dictionary).

# %%
tv = feat.skb.make_learner().with_sklearn_interface()
tv

# %%
# This means our estimator can be used like any other scikit-learn transformer,
# for example in a scikit-learn Pipeline.

# %%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

pipeline = Pipeline([("tv", tv), ("hgb", HistGradientBoostingRegressor())])
pipeline

# %%
cross_validate(
    pipeline, dataset.X, dataset.y, scoring="neg_mean_absolute_percentage_error"
)

# %%
# It can also be used in a scikit-learn
# :class:`~sklearn.model_selection.GridSearchCV` or
# :class:`~sklearn.model_selection.RandomizedSearchCV`.
# :meth:`XySkrubLearner.get_param_grid` prepares for
# us the parameter grid that we can pass to GridSearchCV from the choices in
# the data op. If we provide our own grid for the other steps, skrub takes care of
# combining the 2 grids for us.

# %%
from sklearn.model_selection import GridSearchCV
import pandas as pd

hgb_grid = {"hgb__learning_rate": [0.2, 0.5]}

grid_search = GridSearchCV(
    pipeline, tv.get_param_grid(key_prefix="tv__", outer_grid=hgb_grid)
).fit(dataset.X, dataset.y)

pd.DataFrame(grid_search.cv_results_)[
    ["param_hgb__learning_rate", "param_tv__0:n_components", "mean_test_score"]
].sort_values("mean_test_score", ascending=False)

# %%
# Using the estimator in another DataOp
# -------------------------------------
#
# As our table vectorizer is a normal scikit-learn estimator,
# it can easily be used with :meth:`.skb.apply() <DataOp.skb.apply>`.

# %%
pred = (
    skrub.X()
    .skb.apply(tv)
    .skb.set_name("table_vectorizer")
    .skb.apply(HistGradientBoostingRegressor(), y=skrub.y())
)
pred
