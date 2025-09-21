.. currentmodule:: skrub
.. _user_guide_data_ops_nesting_choices:

Going beyond estimator hyperparameters: nesting choices and choosing pipelines
------------------------------------------------------------------------------

Choices are not limited to scikit-learn hyperparameters: we can use choices
wherever we use DataOps. The choice of the estimator to use, any argument of
an DataOp's method or :func:`deferred` function call, etc. can be replaced
with choices. We can also choose between several DataOps to compare
different pipelines.

As an example of choices outside of scikit-learn estimators, we can consider
several ways to perform an aggregation on a pandas DataFrame:

>>> import skrub
>>> ratings = skrub.var("ratings")
>>> agg_ratings = ratings.groupby("movieId")["rating"].agg(
...     skrub.choose_from(["median", "mean"], name="rating_aggregation")
... )
>>> print(agg_ratings.skb.describe_param_grid())
- rating_aggregation: ['median', 'mean']

We can also choose between several completely different pipelines by turning a
choice into an DataOp, via its ``as_data_op`` method (or by using
:func:`as_data_op` on any object).

>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import Ridge
>>> import skrub
>>> diabetes_df = load_diabetes(as_frame=True)["frame"]
>>> data = skrub.var("data", diabetes_df)
>>> X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
>>> y = data["target"].skb.mark_as_y()

>>> ridge_pred = X.skb.apply(skrub.optional(StandardScaler())).skb.apply(
...     Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α")), y=y
... )
>>> rf_pred = X.skb.apply(
...     RandomForestRegressor(n_estimators=skrub.choose_int(5, 50, name="N 🌴")), y=y
... )
>>> pred = skrub.choose_from({"ridge": ridge_pred, "rf": rf_pred}).as_data_op()
>>> print(pred.skb.describe_param_grid())
- choose_from({'ridge': …, 'rf': …}): 'ridge'
  optional(StandardScaler()): [StandardScaler(), None]
  α: choose_float(0.01, 10.0, log=True, name='α')
- choose_from({'ridge': …, 'rf': …}): 'rf'
  N 🌴: choose_int(5, 50, name='N 🌴')

Also note that as seen above, choices can be nested arbitrarily. For example it
is frequent to choose between several estimators, each of which contains choices
in its hyperparameters.

|


Linking choices depending on other choices
------------------------------------------

Choices can depend on another choice made with :func:`choose_from`,
:func:`choose_bool` or :func:`optional` through those objects' ``.match()``
method.

Suppose we want to use either ridge regression, random forest or gradient
boosting, and that we want to use imputation for ridge and random forest (only),
and scaling for the ridge (only). We can start by choosing the kind of
estimators and make further choices depend on the estimator kind:

>>> import skrub
>>> from sklearn.impute import SimpleImputer, KNNImputer
>>> from sklearn.preprocessing import StandardScaler, RobustScaler
>>> from sklearn.linear_model import Ridge
>>> from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

>>> estimator_kind = skrub.choose_from(
...     ["ridge", "random forest", "gradient boosting"], name="estimator"
... )
>>> imputer = estimator_kind.match(
...     {"gradient boosting": None},
...     default=skrub.choose_from([SimpleImputer(), KNNImputer()], name="imputer"),
... )
>>> scaler = estimator_kind.match(
...     {"ridge": skrub.choose_from([StandardScaler(), RobustScaler()], name="scaler")},
...     default=None,
... )
>>> predictor = estimator_kind.match(
...     {
...         "ridge": Ridge(),
...         "random forest": RandomForestRegressor(),
...         "gradient boosting": HistGradientBoostingRegressor(),
...     }
... )
>>> pred = skrub.X().skb.apply(imputer).skb.apply(scaler).skb.apply(predictor)
>>> print(pred.skb.describe_param_grid())
- estimator: 'ridge'
  imputer: [SimpleImputer(), KNNImputer()]
  scaler: [StandardScaler(), RobustScaler()]
- estimator: 'random forest'
  imputer: [SimpleImputer(), KNNImputer()]
- estimator: 'gradient boosting'

Note that only relevant choices are included in each subgrid. For example, when
the estimator is ``'random forest'``, the subgrid contains several options for
imputation but not for scaling.

In addition to ``match``, choices created with :func:`choose_bool` have an
``if_else()`` method which is a convenience helper equivalent to
``match({True: ..., False: ...})``.
