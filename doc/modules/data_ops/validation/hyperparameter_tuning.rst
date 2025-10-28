.. currentmodule:: skrub
.. _user_guide_data_ops_hyperparameter_tuning:

Using the skrub ``choose_*`` functions to tune hyperparameters
==============================================================

skrub provides a convenient way to declare ranges of possible values, and tune
those choices to keep the values that give the best predictions on a validation
set.

Rather than specifying a grid of hyperparameters separately from the pipeline,
we simply insert special skrub objects in place of the value.

We define the same set of operations as before:

>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import Ridge
>>> import skrub
>>> diabetes_df = load_diabetes(as_frame=True)["frame"]
>>> data = skrub.var("data", diabetes_df)
>>> X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
>>> y = data["target"].skb.mark_as_y()
>>> pred = X.skb.apply(Ridge(), y=y)

Now, we can
replace the hyperparameter ``alpha`` (which should be a float) with a range
created by :func:`skrub.choose_float`. skrub can use it to select the best value
for ``alpha``.



>>> pred = X.skb.apply(
...     Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α")), y=y
... )

.. warning::

   When we do :meth:`.skb.make_learner() <DataOp.skb.make_learner>`, the pipeline
   we obtain does not perform any hyperparameter tuning. The pipeline we obtain
   uses default values for each of the choices. For numeric choices it is the
   middle of the range, and for :func:`choose_from` it is the first option we
   give it.

   To get a pipeline that runs an internal cross-validation to select the best
   hyperparameters, we must use :meth:`.skb.make_grid_search()
   <DataOp.skb.make_grid_search()>` or :meth:`.skb.make_randomized_search()
   <DataOp.skb.make_randomized_search>`.


Here are the different kinds of choices, along with their default outcome when
we are not using hyperparameter search:

.. _choice-defaults-table:

.. list-table:: Default choice outcomes
   :header-rows: 1

   * - Choosing function
     - Description
     - Default outcome
   * - :func:`choose_from([10, 20]) <choose_from>`
     - Choose between the listed options (10 and 20).
     - First outcome in the list: ``10``
   * - :func:`choose_from({"a_name": 10, "b_name": 20}) <choose_from>`
     - Choose between the listed options (10 and 20). Dictionary keys serve as
       names for the options.
     - First outcome in the dictionary: ``10``
   * - :func:`optional(10) <optional>`
     - Choose between the provided value and ``None`` (useful for optional
       transformations in a pipeline, e.g., ``optional(StandardScaler())``).
     - The provided ``value``: ``10``
   * - :func:`choose_bool() <choose_bool>`
     - Choose between True and False.
     - ``True``
   * - :func:`choose_float(1.0, 100.0) <choose_float>`
     - Sample a floating-point number in a range.
     - The middle of the range: ``50.5``
   * - :func:`choose_int(1, 100) <choose_int>`
     - Sample an integer in a range.
     - The integer closest to the middle of the range: ``50``
   * - :func:`choose_float(1.0, 100.0, log=True) <choose_float>`
     - Sample a float in a range on a logarithmic scale.
     - The middle of the range on a log scale: ``10.0``
   * - :func:`choose_int(1, 100, log=True) <choose_int>`
     - Sample an integer in a range on a logarithmic scale.
     - The integer closest to the middle of the range on a log scale: ``10``
   * - :func:`choose_float(1.0, 100.0, n_steps=4) <choose_float>`
     - Sample a float on a grid.
     - The step closest to the middle of the range: ``34.0`` (steps: ``[1.0, 34.0, 67.0, 100.0]``)
   * - :func:`choose_int(1, 100, n_steps=4) <choose_int>`
     - Sample an integer on a grid.
     - The step closest to the middle of the range: ``34`` (steps: ``[1, 34, 67, 100]``)
   * - :func:`choose_float(1.0, 100.0, log=True, n_steps=4) <choose_float>`
     - Sample a float on a logarithmically spaced grid.
     - The step closest to the middle of the range on a log scale: ``4.64``
       (steps: ``[1.0, 4.64, 21.54, 100.0]``)
   * - :func:`choose_int(1, 100, log=True, n_steps=4) <choose_int>`
     - Sample an integer on a logarithmically spaced grid.
     - The step closest to the middle of the range on a log scale: ``5``
       (steps: ``[1, 5, 22, 100]``)


The default choices for an DataOp, those that get used when calling
:meth:`.skb.make_learner() <DataOp.skb.make_learner>`, can be inspected with
:meth:`.skb.describe_defaults() <DataOp.skb.describe_defaults>`:

>>> pred.skb.describe_defaults()
{'α': 0.316...}

We can then find the best hyperparameters.

>>> search = pred.skb.make_randomized_search(fitted=True)
>>> search.results_  # doctest: +SKIP
   mean_test_score         α
0         0.478338  0.141359
1         0.476022  0.186623
2         0.474905  0.205476
3         0.457807  0.431171
4         0.456808  0.443038
5         0.439670  0.643117
6         0.420917  0.866328
7         0.380719  1.398196
8         0.233172  4.734989
9         0.168444  7.780156

Rather than fitting a randomized or grid search to find the best combination, it is also
possible to obtain an iterator over different parameter combinations, to inspect
their outputs or to have manual control over the model selection, using
:meth:`.skb.iter_learners_grid() <DataOp.skb.iter_learners_grid>` or
:meth:`.skb.iter_learners_randomized() <DataOp.skb.iter_learners_randomized>`.
Those yield the candidate pipelines that are explored by the grid and randomized
search respectively.

A human-readable description of parameters for a pipeline can be obtained with
:meth:`SkrubLearner.describe_params`:

>>> search.best_learner_.describe_params() # doctest: +SKIP
{'α': 0.054...}

It is also possible to use :meth:`ParamSearch.plot_results` to visualize the results
of the search using a parallel coordinates plot.

A full example of how to use hyperparameter search is available in
:ref:`sphx_glr_auto_examples_data_ops_13_choices.py`.

|


.. _user_guide_data_ops_feature_selection:

Feature selection with skrub :class:`SelectCols` and :class:`DropCols`
=======================================================================
It is possible to combine :class:`SelectCols` and :class:`DropCols` with
:func:`choose_from` to perform feature selection by dropping specific columns
and evaluating how this affects the downstream performance.

Consider this example. We first define the variable:

>>> import pandas as pd
>>> import skrub.selectors as s
>>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
>>> df = pd.DataFrame({"text": ["foo", "bar", "baz"], "number": [1, 2, 3]})
>>> X = skrub.X(df)

Then, we use the :ref:`skrub selectors <user_guide_selectors>` to encode each
column with a different transformer:

>>> X_enc = X.skb.apply(StandardScaler(), cols=s.numeric()).skb.apply(
...     OneHotEncoder(sparse_output=False), cols=s.string()
... )
>>> X_enc
<Apply OneHotEncoder>
Result:
―――――――
     number  text_bar  text_baz  text_foo
0 -1.224745       0.0       0.0       1.0
1  0.000000       1.0       0.0       0.0
2  1.224745       0.0       1.0       0.0

Now we can use :class:`skrub.DropCols` to define two possible selection strategies:
first, we drop the column ``number``, then we drop all columns that start with
``text``. We rely again on the skrub selectors for this:

>>> from skrub import DropCols
>>> drop = DropCols(cols=skrub.choose_from(
...     {"number": s.cols("number"), "text": s.glob("text_*")})
... )
>>> X_enc.skb.apply(drop)
<Apply DropCols>
Result:
―――――――
   text_bar  text_baz  text_foo
0       0.0       0.0       1.0
1       1.0       0.0       0.0
2       0.0       1.0       0.0

We can see the generated parameter grid with :func:`DataOps.skb.describe_param_grid()`.

>>> X_enc.skb.apply(drop).skb.describe_param_grid()
"- choose_from({'number': …, 'text': …}): ['number', 'text']\n"

A more advanced application of this technique is used in
`this tutorial on forecasting timeseries <https://skrub-data.org/EuroSciPy2025/content/notebooks/single_horizon_prediction.html>`_,
along with the feature engineering required to prepare the columns, and the
analysis of the results.
