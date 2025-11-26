.. currentmodule:: skrub
.. _user_guide_data_ops_tuning_optuna:

.. |make_randomized_search| replace:: :func:`~skrub.DataOp.skb.make_randomized_search`


Tuning skrub DataOps plans with Optuna
=======================================

Optuna is a powerful hyperparameter optimization framework that
can be used to efficiently search for the best hyperparameters for machine
learning models; Optuna includes both sophisticated search algorithms and
tools to monitor and visualize the optimization process.

There are two main ways of using Optuna with skrub DataOps plans: either by using
Optuna as a ``backend`` in the
|make_randomized_search|
method, or by creating an Optuna study and providing it with a skrub
:class:`SkrubLearner`.

.. note::

   To use Optuna with skrub, you need to have Optuna installed in your Python
   environment. You can install it using pip:

   .. code-block:: bash

       pip install optuna


Using Optuna as a backend for randomized search
-------------------------------------------------
The easiest way to use Optuna with skrub is to use it as a backend for
randomized hyperparameter search. This allows us to leverage Optuna's advanced
sampling algorithms and features while keeping the familiar interface of
|make_randomized_search|.

We start by defining a skrub DataOps plan with hyperparameter choices:

>>> import skrub
>>> from sklearn.datasets import make_classification
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.feature_selection import SelectKBest
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.dummy import DummyClassifier

>>> X_a, y_a = make_classification(random_state=0)
>>> X, y = skrub.X(X_a), skrub.y(y_a)
>>> selector = SelectKBest(k=skrub.choose_int(4, 20, log=True, name='k'))
>>> logistic = LogisticRegression(C=skrub.choose_float(0.1, 10.0, log=True, name="C"))
>>> rf = RandomForestClassifier(
...    n_estimators=skrub.choose_int(3, 30, log=True, name="N 🌴"),
...    random_state=0,
... )
>>> classifier = skrub.choose_from(
... {"logistic": logistic, "rf": rf, "dummy": DummyClassifier()}, name="classifier"
... )
>>> pred = X.skb.apply(selector, y=y).skb.apply(classifier, y=y)
>>> print(pred.skb.describe_param_grid())
- k: choose_int(4, 20, log=True, name='k')
  classifier: 'logistic'
  C: choose_float(0.1, 10.0, log=True, name='C')
- k: choose_int(4, 20, log=True, name='k')
  classifier: 'rf'
  N 🌴: choose_int(3, 30, log=True, name='N 🌴')
- k: choose_int(4, 20, log=True, name='k')
  classifier: 'dummy'


Now, we can create a randomized search using Optuna as the backend:

>>> search = pred.skb.make_randomized_search(fitted=True, random_state=0, backend="optuna") # doctest: +SKIP

It's possible to access the same parameters as the default backend:

>>> search.results_  # doctest: +SKIP
k         C   N 🌴 classifier  mean_test_score
0   4       NaN   6.0         rf             0.93
1   4  0.645966   NaN   logistic             0.92
2   4       NaN   4.0         rf             0.92
3   4       NaN   3.0         rf             0.90
4   8       NaN  11.0         rf             0.88
5   9       NaN  11.0         rf             0.88
6  14  0.391899   NaN   logistic             0.81
7  19       NaN   4.0         rf             0.72
8  20       NaN   NaN      dummy             0.50
9   9       NaN   NaN      dummy             0.50

The best learner and best hyperparameters can be accessed as usual:

>>> search.best_learner_.describe_params()  # doctest: +SKIP
{'k': 4, 'N 🌴': 6, 'classifier': 'rf'}

|make_randomized_search|
accepts ``sampler`` and ``timeout`` parameters to customize the Optuna study.
Optuna studies feature a wide range of additional parameters, which can be accessed
by using Optuna directly with skrub learners, as shown in the next section.

A more complete example that includes more advanced usage is available in
:ref:`example_optuna_choices`.

Setting a storage for the Optuna study
-------------------------------------------------
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

By default, no storage is used, and the study is kept in memory only.

Using Optuna directly with skrub learners
-------------------------------------------------
It is also possible to use Optuna directly with skrub learners. This allows for more
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

Then, we can inspect the parameters as usual:

>>> best_learner.describe_params() # doctest: +SKIP
{'k': 4, 'C': 0.3031965763542701, 'classifier': 'logistic'}

You can find a more complete example in :ref:`example_optuna_choices`.


Parallelism with Optuna
-------------------------------------------------
:meth:`skrub.cross_validate` and :meth:`optuna.study.Study.optimize` both
accept parameters to control parallelism, however :meth:`skrub.cross_validate`
uses joblib for parallelism, which implmenents process-based parallelism, while
Optuna uses multi-threading for parallelism.

:meth:`skrub.cross_validate` parallelizes by evaluating different splits
in parallel, while :meth:`optuna.study.Study.optimize` parallelizes by
evaluating different trials in parallel. Depending on the use case, multi-processing
may be preferred (e.g., when the evaluation of a single split is very expensive),
or multi-threading (e.g., when the overhead of process-based parallelism is
too high compared to the evaluation time of a single split).

In |make_randomized_search|,
the ``n_jobs`` parameter can be set to 1 to disable joblib parallelism, allowing
Optuna's multi-threading parallelism to be used without interference. Additionally,
multi-threading parallelism is enabled by default if the ``timeout`` parameter
is set.


Using the Optuna dashboard
-------------------------------------------------
Optuna provides a dashboard called Optuna Dashboard that allows us to visualize
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
