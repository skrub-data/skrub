.. currentmodule:: skrub
.. _user_guide_data_ops_tuning_optuna:

.. |make_randomized_search| replace:: :func:`~skrub.DataOp.skb.make_randomized_search`


Tuning DataOps with Optuna
==========================

Optuna is a powerful hyperparameter optimization framework that
can be used to efficiently search for the best hyperparameters for machine
learning models; Optuna includes both sophisticated search algorithms and
tools to monitor and visualize the optimization process.

There are two main ways of using Optuna with skrub DataOps: either by using
Optuna as a ``backend`` in the
|make_randomized_search|
method, or by creating an Optuna study directly and using it to pick values for
skrub choices when calling :meth:`DataOp.skb.make_learner()`.

.. note::

   To use Optuna with skrub, you need to have Optuna installed in your Python
   environment. You can install it using pip:

   .. code-block:: bash

       pip install optuna


Using Optuna as a backend for randomized search
-------------------------------------------------
The easiest way to use Optuna with skrub is to use it as a backend for
|make_randomized_search|. This allows us to leverage Optuna's advanced
sampling algorithms and features while keeping same the familiar interface as
for other search methods.

We start by defining a DataOp containing choices:

>>> import skrub
>>> from sklearn.datasets import make_classification
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.feature_selection import SelectKBest
>>> from sklearn.ensemble import HistGradientBoostingClassifier
>>> from sklearn.dummy import DummyClassifier

>>> X_a, y_a = make_classification(random_state=0)
>>> X, y = skrub.X(X_a), skrub.y(y_a)
>>> selector = SelectKBest(k=skrub.choose_int(4, 20, log=True, name='k'))
>>> logistic = LogisticRegression(C=skrub.choose_float(0.1, 10.0, log=True, name="C"))
>>> hgb = HistGradientBoostingClassifier(
...    learning_rate=skrub.choose_float(.01, .5, log=True, name="learning_rate"),
...    random_state=0,
... )
>>> classifier = skrub.choose_from(
... {"logistic": logistic, "hgb": hgb, "dummy": DummyClassifier()}, name="classifier"
... )
>>> pred = X.skb.apply(selector, y=y).skb.apply(classifier, y=y)
>>> print(pred.skb.describe_param_grid())
- k: choose_int(4, 20, log=True, name='k')
  classifier: 'logistic'
  C: choose_float(0.1, 10.0, log=True, name='C')
- k: choose_int(4, 20, log=True, name='k')
  classifier: 'hgb'
  learning_rate: choose_float(0.01, 0.5, log=True, name='learning_rate')
- k: choose_int(4, 20, log=True, name='k')
  classifier: 'dummy'


Now, we can create a randomized search using Optuna as the backend:

>>> search = pred.skb.make_randomized_search(fitted=True, random_state=0, backend="optuna") # doctest: +SKIP
Running optuna search for study skrub_randomized_search_c4af73b2-45fb-49ca-9f06-092d74aa8118 in storage .../tmpuor7hqjm_skrub_optuna_search_storage/optuna_storage

It is possible to access the same parameters as with the default backend:

>>> search.results_  # doctest: +SKIP
    k         C  learning_rate classifier  mean_test_score
0   4       NaN       0.013146        hgb             0.93
1   4       NaN       0.040454        hgb             0.92
2  19       NaN       0.019968        hgb             0.92
3   4  0.645966            NaN   logistic             0.92
4   4       NaN       0.023337        hgb             0.92
5   8       NaN       0.097994        hgb             0.90
6   9       NaN       0.104104        hgb             0.88
7  14  0.391899            NaN   logistic             0.81
8  20       NaN            NaN      dummy             0.50
9   9       NaN            NaN      dummy             0.50

The best learner and best hyperparameters can be accessed as usual:

>>> search.best_learner_.describe_params()  # doctest: +SKIP
{'k': 4, 'learning_rate': 0.01314593370942781, 'classifier': 'hgb'}

|make_randomized_search|
accepts ``sampler`` and ``timeout`` parameters to customize the Optuna study.
Optuna studies feature a wide range of additional parameters, which can be accessed
by using Optuna directly with skrub learners, as shown in the next section.

A more complete example that includes more advanced usage is available in
:ref:`example_optuna_choices`.

Setting a storage for the Optuna study
--------------------------------------
When using Optuna as a backend for hyperparameter search, it is possible to
specify a storage option to persist the study and its results. This allows us to
resume the search later or analyze the results after the search is complete.
This can be done by providing the ``storage`` parameter to
|make_randomized_search|.

.. code-block:: python

    search = pred.skb.make_randomized_search(
        fitted=True,
        random_state=0,
        backend="optuna",
        storage="sqlite:///optuna_study.db",  # Use a SQLite database file
    )

If no storage is provided, a temporary storage is used during optimization, then
the study is moved to an in-memory storage once the search completes so the
resulting search object is self-contained.

Using Optuna directly
---------------------
It is also possible to use Optuna directly with skrub DataOps. This allows for more
flexibility and control over the optimization process, as we can define custom
objectives and leverage Optuna's advanced features, such as the ask-and-tell interface,
trial pruning, and multi-objective optimization.

In this case, rather than running the hyperparameter search through
|make_randomized_search|,
the :class:`optuna.Study <optuna.study.Study>` runs the hyperparameter
search by defining an objective function that uses a skrub
learner with hyperparameters suggested by Optuna.

:meth:`optimize <optuna.study.Study.optimize>` is given an
``objective`` function. The ``objective`` must accept a
:class:`~optuna.trial.Trial` object (which is produced by the study and picks
the parameters for a given evaluation of the objective) and return the value
to maximize (or minimize).

To use Optuna with a :class:`DataOp`, we just need to pass the Trial object
to :meth:`DataOp.skb.make_learner`. This creates a :class:`SkrubLearner`
initialized with the parameters picked by the optuna Trial.

We can then cross-validate the:class:`SkrubLearner`, or score it however we prefer,
and return the score so that the optuna Study can take it into account.

Here we return a single score (R²), but multi-objective
optimization is also possible. Please refer to the Optuna documentation for
more information.

>>> import optuna # doctest: +SKIP

>>> def objective(trial): # doctest: +SKIP
...    learner = pred.skb.make_learner(choose=trial)
...    cv_results = skrub.cross_validate(learner, environment=pred.skb.get_data(), cv=4)
...    return cv_results["test_score"].mean()

>>> study = optuna.create_study(direction="maximize") # doctest: +SKIP
>>> study.optimize(objective, n_trials=16) # doctest: +SKIP
>>> best_params = study.best_params # doctest: +SKIP

Then, we can create the best learner using the best trial found by Optuna:

>>> best_learner = pred.skb.make_learner(choose=study.best_trial) # doctest: +SKIP

The learner can also be defined as follows:

>>> best_learner = pred.skb.make_learner() # doctest: +SKIP
>>> best_learner.set_params(**study.best_params) # doctest: +SKIP
SkrubLearner(data_op=<Apply HistGradientBoostingClassifier>)

Then, we can inspect the parameters as usual:

>>> best_learner.describe_params() # doctest: +SKIP
{'k': 12, 'learning_rate': 0.06401143720094754, 'classifier': 'hgb'}

You can find a more complete example in :ref:`example_optuna_choices`.


Parallelism
-----------

Optuna's :meth:`optuna.study.Study.optimize` uses thread-based parallelism. When
we use :meth:`DataOp.skb.make_randomized_search` with the Optuna backend, both
threading and multiprocessing can be used. Skrub will choose based on the joblib
configuration: if joblib is configured to use processes (the default),
parallelization is done with joblib, and if joblib is configured to use the
"threading" backend, Optuna's built-in thread-based parallelism is used instead.

When the ``timeout`` parameter is used, Optuna's built-in, thread-based
parallelization is always used regardless of the joblib configuration.


Using the Optuna dashboard
--------------------------
Optuna provides a dashboard that allows us to visualize
and monitor the optimization process in real-time. This can be especially useful
for long-running hyperparameter searches.
To use the Optuna Dashboard, we need to install it first:

.. code-block:: bash

    pip install optuna-dashboard

We can then start the dashboard by running the following command in the terminal:

.. code-block:: bash

    optuna-dashboard STORAGE_URL

Where ``STORAGE_URL`` is the same storage URL used in the Optuna study.

We can then access the dashboard in our web browser at
``http://localhost:8080`` (by default). The dashboard provides various visualizations
and tools to analyze the optimization process, such as parameter importance,
optimization history, and parallel coordinate plots.
