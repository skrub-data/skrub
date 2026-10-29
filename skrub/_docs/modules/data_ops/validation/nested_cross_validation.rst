.. currentmodule:: skrub
.. _user_guide_data_ops_nested_cross_validation:

Validating hyperparameter search with nested cross-validation
=============================================================

To avoid overfitting hyperparameters, the best combination must be evaluated on
data that has not been used to select hyperparameters. This can be done with a
single train-test split or with nested cross-validation.

Using the same examples as the previous sections:

>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import Ridge
>>> import skrub
>>> diabetes_df = load_diabetes(as_frame=True)["frame"]
>>> data = skrub.var("data", diabetes_df)
>>> X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
>>> y = data["target"].skb.mark_as_y()
>>> pred = X.skb.apply(
...     Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α")), y=y
... )

Single train-test split:

>>> split = pred.skb.train_test_split()
>>> search = pred.skb.make_randomized_search()
>>> search.fit(split['train'])
ParamSearch(data_op=<Apply Ridge>,
            search=RandomizedSearchCV(estimator=None, param_distributions=None))
>>> search.score(split['test'])  # doctest: +SKIP
0.4922874902029253

For nested cross-validation we use :func:`skrub.cross_validate`, which accepts a
``pipeline`` parameter (as opposed to
:meth:`.skb.cross_validate() <DataOp.skb.cross_validate>`
which always uses the default hyperparameters):

>>> skrub.cross_validate(pred.skb.make_randomized_search(), pred.skb.get_data())  # doctest: +SKIP
   fit_time  score_time  test_score
0  0.891390    0.002768    0.412935
1  0.889267    0.002773    0.519140
2  0.928562    0.003124    0.491722
3  0.890453    0.002732    0.428337
4  0.889162    0.002773    0.536168
