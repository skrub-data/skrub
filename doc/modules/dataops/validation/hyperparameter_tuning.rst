.. _hyperparameter_tuning:

Using the Skrub ``choose_*`` functions to tune hyperparameters
==============================================================

Skrub provides a convenient way to declare ranges of possible values, and tune
those choices to keep the values that give the best predictions on a validation
set.

Rather than specifying a grid of hyperparameters separately from the pipeline,
we simply insert special skrub objects in place of the value. For example we
replace the hyperparameter ``alpha`` (which should be a float) with a range
created by :func:`skrub.choose_float`. Skrub can use it to select the best value
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
     - choose between the listed options 10 and 20
     - first outcome in the list: ``10``
   * - :func:`choose_from({"a_name": 10, "b_name": 20}) <choose_from>`
     - choose between the listed options 10 and 20, dictionary keys are names
       for the options.
     - first outcome in the dict: ``10``
   * - :func:`optional(10) <optional>`
     - choose between the provided value and ``None`` (useful for optional
       transformations in a pipeline eg ``optional(StandardScaler())``).
     - the provided ``value``: ``10``
   * - :func:`choose_bool() <choose_bool>`
     - choose between True and False.
     - ``True``
   * - :func:`choose_float(1.0, 100.0) <choose_float>`
     - sample a floating-point number in a range.
     - the middle of the range: ``50.5``
   * - :func:`choose_int(1, 100) <choose_int>`
     - sample an integer in a range.
     - the int closest to the middle of the range: ``50``
   * - :func:`choose_float(1.0, 100.0, log=True) <choose_float>`
     - sample a float in a range on a logarithmic scale.
     - the middle of the range on a log scale: ``10.0``
   * - :func:`choose_int(1, 100, log=True) <choose_int>`
     - sample an int in a range on a logarithmic scale.
     - the int closest to the middle of the range on a log scale: ``10``
   * - :func:`choose_float(1.0, 100.0, n_steps=4) <choose_float>`
     - sample a float on a grid.
     - the step closest to the middle of the range: ``34.0`` (here steps are
       ``[1.0, 34.0, 67.0, 100.0]``)
   * - :func:`choose_int(1, 100, n_steps=4) <choose_int>`
     - sample an int on a grid.
     - the (integer) step closest to the middle of the range: ``34`` (here steps are
       ``[1, 34, 67, 100]``)
   * - :func:`choose_float(1.0, 100.0, log=True, n_steps=4) <choose_float>`
     - sample a float on a logarithmically-spaced grid.
     - the step closest to the middle of the range on a log scale: ``4.64``
       (here steps are ``[1.0, 4.64, 21.54, 100.0]``)
   * - :func:`choose_int(1, 100, log=True, n_steps=4) <choose_int>`
     - sample an int on a logarithmically-spaced grid.
     - the step closest to the middle of the range on a log scale: ``5``
       (here steps are ``[1, 5, 22, 100]``)
     - the (integer) step closest to the middle of the range on a log scale: ``5``
       (here steps are ``[1, 5, 22, 100]``)


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
:ref:`sphx_glr_auto_examples_data_ops_12_choices.py`.
