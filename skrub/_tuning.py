"""
Hyperparameter grids
====================

This module provides the functionality that allows specifying ranges of
hyperparameters directly inside the estimators when adding steps to a
skrub expression. They are the low-level building blocks that allow users to
write code like:

>>> X.apply(choose_from([PCA(), SelectKBest()]), y=y)  # doctest: +SKIP
>>> X.apply(Ridge(alpha=choose_float(0.01, 100.0, log=True)), y=y)  # doctest: +SKIP

The main components are classes that represent ranges of hyperparameters, such
as ``Choice``, and the ``expand_grid`` function, which inspects a pipeline
containing choices and constructs a hyperparameter grid suitable for
scikit-learn's ``GridSearchCV`` or ``RandomizedSearchCV``.

To illustrate what ``expand_grid`` does, let us construct by hand a
hyperparameter grid for a simple pipeline.

Suppose we have a pipeline with a dimensionality reduction step and a
regressor, which we might construct like this:

>>> from sklearn.pipeline import Pipeline
>>> from sklearn.decomposition import PCA
>>> from sklearn.linear_model import Ridge

>>> model = Pipeline([("dim_reduction", PCA()), ("regressor", Ridge())])

If we want to tune the choice of estimators or their hyperparameters, we need to
construct a grid of possible values for each estimator and its hyperparameters.
Then the ``GridSearchCV`` will evaluate each possible combination of values and
select the best.

Let us start with a very simple grid that contains only one node: the estimators
we have used above and their default parameters.

>>> grid = {"dim_reduction": [PCA()], "regressor": [Ridge()]}

Note here that ``[PCA()]`` is wrapped in a list: it is the list of all
dimensionality reduction transformers we want to include in the grid. Here we
have only one option for each item so our grid has 1 x 1 = 1 node.

We now add a range of values for the ``alpha`` hyperparameter of the ridge:

>>> grid = {
...     "dim_reduction": [PCA()],
...     "regressor": [Ridge()],
...     "regressor__alpha": [0.1, 1.0, 10.0],
... }

We may also want to try different estimators, rather than just modifying the
parameters of a given estimator. Suppose we want to consider ``SelectKBest``
as an alternative to the ``PCA``:

>>> from sklearn.feature_selection import SelectKBest, f_regression

>>> grid = {
...     "dim_reduction": [PCA(), SelectKBest(f_regression)],
...     "regressor": [Ridge()],
...     "regressor__alpha": [0.1, 1.0, 10.0],
... }

Now we want to specify ranges for the parameters that control the reduced
dimension: ``n_components`` for the ``PCA`` and ``k`` for ``SelectKBest``. Note
they are incompatible: setting ``k`` on a ``PCA`` would be an error, so we need
to separate them by splitting our grid into 2 subgrids. Instead of one
dictionary we now have a list containing 2 dictionaries.

>>> grid = [
...     {
...         "dim_reduction": [PCA()],
...         "regressor": [Ridge()],
...         "dim_reduction__n_components": [10, 20, 30],
...         "regressor__alpha": [0.1, 1.0, 10.0],
...     },
...     {
...         "dim_reduction": [SelectKBest(f_regression)],
...         "regressor": [Ridge()],
...         "dim_reduction__k": [10, 20, 30],
...         "regressor__alpha": [0.1, 1.0, 10.0],
...     },
... ]

This is an important step that the ``expand_grid`` function of this module has
to perform: whenever we have a choice of different estimators, and one of those
estimators has a choice of different values for one of its hyperparameters, the
grid must be split. The estimator must be placed into its own subgrid which
contains its ranges of hyperparameters.

If we also want alternatives to the ``Ridge``, and if we have nested estimators
such as a ``StackingRegressor`` or ``TableVectorizer``, constructing such
hyperparameter grids by hand becomes unwieldy. Moreover, here we are
constructing and storing the grid separately from the pipeline, which is
error-prone and impractical.

This module allows us to store ranges of values (possibly nested) directly in
the estimators. Then, ``expand_grid`` will extract them and construct the grid
and its subgrids as needed.

We call a "choice" a range of things that we can pick from, such as estimators
or hyperparameter values. Each of those things —the result of choosing— is
referred to as an "outcome".

We have different kinds of choices for enumerated sets and numerical ranges. The
simplest one is an explicitly enumerated set of discrete options, constructed
with ``choose_from``. Other options are described later.

Let us rebuild the previous grid using ``choose_from``. To make things more
interesting we also add a ``LinearSVR`` as an alternative to the ``Ridge``.

>>> from sklearn.svm import LinearSVR
>>> from skrub._tuning import choose_from

>>> grid = {
...     "dim_reduction": choose_from([PCA(), SelectKBest(f_regression)]),
...     "regressor": choose_from([Ridge(), LinearSVR()]),
... }

We can also add hyperparameter ranges. The key point here is that choices can be
nested, rather than manually extracting them with names such as
``regressor__alpha`` and building a flat list of subgrids.

>>> n_dim = choose_from([10, 20, 30])
>>> pca = PCA(n_components=n_dim)
>>> kbest = SelectKBest(f_regression, k=n_dim)

>>> ridge = Ridge(alpha=choose_from([0.1, 1.0, 10.0]))
>>> svr = LinearSVR(C=choose_from([0.1, 1.0]))

>>> grid = {
...     "dim_reduction": choose_from([pca, kbest]),
...     "regressor": choose_from([ridge, svr]),
... }
>>> grid
{'dim_reduction': choose_from([PCA(n_components=choose_from([10, 20, 30])), SelectKBest(k=choose_from([10, 20, 30]),
            score_func=<function f_regression at 0x...>)]), 'regressor': choose_from([Ridge(alpha=choose_from([0.1, 1.0, 10.0])), LinearSVR(C=choose_from([0.1, 1.0]))])}

Now we have a few nested choices that do not have any meaning for scikit-learn.
In order to obtain a grid that can be used with ``GridSearchCV``, we need to
extract the necessary information with ``expand_grid``.

>>> from skrub._tuning import expand_grid
>>> from pprint import pprint

>>> expanded = expand_grid(grid)
>>> pprint(expanded)
[{'dim_reduction': choose_from([PCA(n_components=<dim_reduction__n_components>)]),
  'dim_reduction__n_components': choose_from([10, 20, 30]),
  'regressor': choose_from([Ridge(alpha=<regressor__alpha>)]),
  'regressor__alpha': choose_from([0.1, 1.0, 10.0])},
 {'dim_reduction': choose_from([PCA(n_components=<dim_reduction__n_components>)]),
  'dim_reduction__n_components': choose_from([10, 20, 30]),
  'regressor': choose_from([LinearSVR(C=<regressor__C>)]),
  'regressor__C': choose_from([0.1, 1.0])},
 {'dim_reduction': choose_from([SelectKBest(k=<dim_reduction__k>,
            score_func=<function f_regression at 0x...>)]),
  'dim_reduction__k': choose_from([10, 20, 30]),
  'regressor': choose_from([Ridge(alpha=<regressor__alpha>)]),
  'regressor__alpha': choose_from([0.1, 1.0, 10.0])},
 {'dim_reduction': choose_from([SelectKBest(k=<dim_reduction__k>,
            score_func=<function f_regression at 0x...>)]),
  'dim_reduction__k': choose_from([10, 20, 30]),
  'regressor': choose_from([LinearSVR(C=<regressor__C>)]),
  'regressor__C': choose_from([0.1, 1.0])}]

``expand_grid`` has constructed a hyperparameter grid with 4 subgrids which can
be fed to ``GridSearchCV`` or ``RandomizedSearchCV``. (The ``choose_from``
objects implement the interface of a sequence so they look like lists to
``GridSearchCV``, which is what it expects).

To inspect a grid a bit more easily we have the convenience function ``grid_description``.

>>> from skrub._tuning import grid_description

>>> print(grid_description(expanded))
- 'dim_reduction': PCA(n_components=<dim_reduction__n_components>)
  'dim_reduction__n_components':
      - 10
      - 20
      - 30
  'regressor': Ridge(alpha=<regressor__alpha>)
  'regressor__alpha':
      - 0.1
      - 1.0
      - 10.0
- 'dim_reduction': PCA(n_components=<dim_reduction__n_components>)
  'dim_reduction__n_components':
      - 10
      - 20
      - 30
  'regressor': LinearSVR(C=<regressor__C>)
  'regressor__C':
      - 0.1
      - 1.0
- 'dim_reduction': SelectKBest(k=<dim_reduction__k>,
                               score_func=<function f_regression at 0x...>)
  'dim_reduction__k':
      - 10
      - 20
      - 30
  'regressor': Ridge(alpha=<regressor__alpha>)
  'regressor__alpha':
      - 0.1
      - 1.0
      - 10.0
- 'dim_reduction': SelectKBest(k=<dim_reduction__k>,
                               score_func=<function f_regression at 0x...>)
  'dim_reduction__k':
      - 10
      - 20
      - 30
  'regressor': LinearSVR(C=<regressor__C>)
  'regressor__C':
      - 0.1
      - 1.0

Here we see that we get the default labels for the choices (such as
``'regressor__alpha'``) and their outcomes (such as
``'Ridge(alpha=<regressor__alpha>)'``). However ``choose_from`` allows us to
provide human-readable labels, both for the choice itself and for each
individual outcome.

We can rewrite our grid with labels we choose.

Giving a name to a choice:

>>> n_dim = choose_from([10, 20, 30], name="n dimensions")
>>> pca = PCA(n_components=n_dim)
>>> kbest = SelectKBest(f_regression, k=n_dim)

>>> ridge = Ridge(alpha=choose_from([0.1, 1.0, 10.0], name="α"))
>>> svr = LinearSVR(C=choose_from([0.1, 1.0], name="C"))

Giving names to the outcomes is also possible, by passing a dictionary rather
than a list to ``choose_from``. The dictionary keys are the names of the
corresponding values:

>>> grid = {
...     "dim_reduction": choose_from({"pca": pca, "kbest": kbest}),
...     "regressor": choose_from({"ridge": ridge, "svr": svr}),
... }
>>> expanded = expand_grid(grid)

Let us display the new grid:

>>> print(grid_description(expanded))
- 'dim_reduction': 'pca'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'ridge'
  'α':
      - 0.1
      - 1.0
      - 10.0
- 'dim_reduction': 'pca'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'svr'
  'C':
      - 0.1
      - 1.0
- 'dim_reduction': 'kbest'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'ridge'
  'α':
      - 0.1
      - 1.0
      - 10.0
- 'dim_reduction': 'kbest'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'svr'
  'C':
      - 0.1
      - 1.0

This makes our grid description more readable, but most importantly it allows
the ``Recipe`` to use those human-readable labels and provide a much better
display when showing hyperparameter search results (either in a table or in the
parallel coordinate plots).

Finally, let us note that choices can be nested as deep as we want for example
in meta-estimators. Imagine we also want to consider stacking as an alternative
to ridge and the SVR.

>>> from sklearn.ensemble import BaggingRegressor

>>> bagging = BaggingRegressor(
...     estimator=choose_from({'ridge': ridge, 'svr': svr}, name="bagged estimator"),
...     n_estimators=choose_from([10, 20], name="n bagged estimators")
... )
>>> grid = {
...     "dim_reduction": choose_from({"pca": pca, "kbest": kbest}),
...     "regressor": choose_from({"ridge": ridge, "svr": svr, "bagging": bagging}),
... }
>>> expanded = expand_grid(grid)
>>> print(grid_description(expanded))
- 'dim_reduction': 'pca'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'ridge'
  'α':
      - 0.1
      - 1.0
      - 10.0
- 'dim_reduction': 'pca'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'svr'
  'C':
      - 0.1
      - 1.0
- 'dim_reduction': 'pca'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'bagging'
  'bagged estimator': 'ridge'
  'α':
      - 0.1
      - 1.0
      - 10.0
  'n bagged estimators':
      - 10
      - 20
- 'dim_reduction': 'pca'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'bagging'
  'bagged estimator': 'svr'
  'C':
      - 0.1
      - 1.0
  'n bagged estimators':
      - 10
      - 20
- 'dim_reduction': 'kbest'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'ridge'
  'α':
      - 0.1
      - 1.0
      - 10.0
- 'dim_reduction': 'kbest'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'svr'
  'C':
      - 0.1
      - 1.0
- 'dim_reduction': 'kbest'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'bagging'
  'bagged estimator': 'ridge'
  'α':
      - 0.1
      - 1.0
      - 10.0
  'n bagged estimators':
      - 10
      - 20
- 'dim_reduction': 'kbest'
  'n dimensions':
      - 10
      - 20
      - 30
  'regressor': 'bagging'
  'bagged estimator': 'svr'
  'C':
      - 0.1
      - 1.0
  'n bagged estimators':
      - 10
      - 20

Choices and Outcomes
====================

A "choice" represents a range of things from which we can choose. Each of those
things --the result of choosing-- is referred to here as an "outcome".

Some hyperparameters take their value from a discrete, explicitly enumerated set
(for example the ``svd_solver`` of a ``PCA`` can be "auto", "full",
"covariance_eigh", ...). Some others take their values from a range of real or
integral numbers (for example the ``n_components`` of a ``PCA`` is an int
between 0 and the smallest dimension of ``X``). This module allows to represent
all those kinds of ranges.

The simplest kind is the enumerated choice we have seen before.

>>> dim_reduction = choose_from([PCA(), SelectKBest()])
>>> dim_reduction
choose_from([PCA(), SelectKBest()])
>>> type(dim_reduction)
<class 'skrub._tuning.Choice'>
>>> dim_reduction.outcomes
[Outcome(value=PCA(), name=None, in_choice=None), Outcome(value=SelectKBest(), name=None, in_choice=None)]

``name`` and ``in_choice`` in the outcomes record the labels given to the
outcome itself and the ``Choice`` that contains it, which we have not specified
here. We can provide them which is useful to the ``Recipe`` for displaying
search results in a table or a plot.

>>> dim_reduction = choose_from({'pca': PCA(), 'kbest': SelectKBest()}, name='dim reduction')
>>> dim_reduction
choose_from({'pca': PCA(), 'kbest': SelectKBest()}, name='dim reduction')
>>> dim_reduction.outcomes
[Outcome(value=PCA(), name='pca', in_choice='dim reduction'), Outcome(value=SelectKBest(), name='kbest', in_choice='dim reduction')]

The helper ``unwrap`` extracts the value from its argument if it is an outcome,
otherwise returns the argument:

>>> unwrap(dim_reduction.outcomes[0])
PCA()
>>> unwrap(PCA())
PCA()

The ``Choice`` is a sequence, which allows it to be used in a hyperparameter
grid passed to a ``GridSearchCV``:

>>> list(dim_reduction)
[Outcome(value=PCA(), name='pca', in_choice='dim reduction'), Outcome(value=SelectKBest(), name='kbest', in_choice='dim reduction')]

Choices also have a default outcome, which is used if we want to get a pipeline
and fit it without doing any hyperparameter search (e.g. for the previews of the
``Recipe``). For enumerated choices, the default outcome is simply the first in
the list:

>>> dim_reduction.default()
Outcome(value=PCA(), name='pca', in_choice='dim reduction')

Note: in the future, we may want to allow providing an explicit default
(especially for numeric choices, described in the next section).

We can use the ``unwrap_default`` helper to get the default outcome and extract
its ``value``:

>>> from skrub._tuning import unwrap_default
>>> unwrap_default(dim_reduction)
PCA()

Like ``unwrap``, it returns its argument when it is just a regular value:

>>> unwrap_default(PCA())
PCA()

Numeric choices
---------------

A ``NumericChoice`` is a range of numbers between a low and a high value, either
on a log or linear scale. It can produce either floats and ints. It exposes a
``rvs`` method to draw a random sample from the range, making it possible to use
with scikit-learn's ``RandomizedSearchCV``. It can be constructed with
``choose_int`` or ``choose_float``.

>>> from skrub._tuning import choose_int, choose_float
>>> n_dims = choose_int(10, 100, name='n dims')
>>> n_dims
choose_int(10, 100, name='n dims')
>>> type(n_dims)
<class 'skrub._tuning.NumericChoice'>
>>> n_dims.rvs() # doctest: +SKIP
NumericOutcome(value=98, name=None, in_choice='n dims', is_from_log_scale=False)

If we would rather use a log scale:

>>> n_dims = choose_int(10, 100, name='n dims', log=True)
>>> n_dims
choose_int(10, 100, log=True, name='n dims')
>>> n_dims.rvs() # doctest: +SKIP
NumericOutcome(value=80, name=None, in_choice='n dims', is_from_log_scale=True)

As we can see, the outcomes record whether they came from a log scale, which is
useful to set the scales of axes in the plots where they are displayed.

We can illustrate the difference between log and linear scale by drawing a
larger sample:

>>> n = choose_int(0, 100, log=False)
>>> q = [0, 25, 50, 75, 100]
>>> np.percentile([unwrap(v) for v in n.rvs(1000, random_state=0)], q)
array([ 0., 24., 48., 73., 99.])
>>> n = choose_int(1, 100, log=True)
>>> np.percentile([unwrap(v) for v in n.rvs(1000, random_state=0)], q)
array([ 1.,  3.,  9., 29., 99.])

If we need floating-point numbers rather than ints we use ``choose_float``:

>>> alpha = choose_float(.01, 100, log=True, name='α')
>>> alpha
choose_float(0.01, 100, log=True, name='α')
>>> alpha.rvs() # doctest: +SKIP
NumericOutcome(value=16.656593316727974, name=None, in_choice='α', is_from_log_scale=True)

The default outcome of a numeric choice is the middle of its range (either on a
linear or log scale):

>>> unwrap_default(choose_float(0.0, 100.0, log=False)) # doctest: +SKIP
50.0
>>> unwrap_default(choose_float(1.0, 100.0, log=True)) # doctest: +SKIP
10.000000000000002

It is possible to specify a number of steps on the range of value to discretize
it. Too big a step size loses the benefits of a randomized search. However,
discretizing the range is useful to make better use of caching by reusing the
same values. If we set a number of steps, we get a ``DiscretizedNumericChoice``.

>>> n_dims = choose_int(10, 100, name='n dims', log=True, n_steps=5)
>>> n_dims
choose_int(10, 100, log=True, n_steps=5, name='n dims')
>>> type(n_dims)
<class 'skrub._tuning.DiscretizedNumericChoice'>

Discretized ranges are sequences so they can be used either with
``RandomizedSearchCV`` or ``GridSearchCV``.

>>> n_dims.rvs() # doctest: +SKIP
NumericOutcome(value=32, name=None, in_choice='n dims', is_from_log_scale=True)
>>> list(n_dims) # doctest: +SKIP
[10, 18, 32, 56, 100]

The default outcome is as close as possible to the middle while respecting the
steps:

>>> unwrap_default(n_dims) # doctest: +SKIP
32

Optional
--------

Finally, it is common to choose between something and nothing. The last type of
choice is a shorthand for a choice between a given value and ``None``:

>>> from skrub._tuning import optional
>>> feature_selection = optional(SelectKBest(), name='feature selection')
>>> feature_selection
optional(SelectKBest(), name='feature selection')
>>> type(feature_selection)
<class 'skrub._tuning.Optional'>
>>> list(feature_selection)
[Outcome(value=SelectKBest(), name='true', in_choice='feature selection'), Outcome(value=None, name='false', in_choice='feature selection')]
>>> unwrap_default(feature_selection)
SelectKBest()
"""  # noqa: E501

import dataclasses
import io
import typing
from collections.abc import Sequence

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.utils import check_random_state

from . import _utils

#
# Choices and Outcomes
# ====================
#
# We start by defining the different kinds of choices and their outcomes, and
# the functions that construct them such as ``choose_from`` and
# ``choose_float``.
#


def _with_fields(obj, **fields):
    """
    Make a copy of a dataclass instance with different values for some of the attributes
    """
    return obj.__class__(
        **({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)} | fields)
    )


@dataclasses.dataclass
class Outcome:
    """The outcome of a choice.

    Represents one of the different options that constitute a choice. It
    records its value, an optional name for the outcome (such as "ridge" and
    "lasso" in

    ``choose_from({"ridge": Ridge(), "lasso": Lasso()}, name="regressor")``)

    and optionally the name of the choice it belongs to (such as "regressor" in
    the example above).
    """

    value: typing.Any
    name: typing.Optional[str] = None
    in_choice: typing.Optional[str] = None

    def __str__(self):
        if self.name is not None:
            return repr(self.name)
        return repr(self.value)


class BaseChoice:
    """A base class for all kinds of choices (enumerated, numeric range, ...)

    The main reason they all derive from this base class is to make them easily
    recognizable with ``isinstance`` checks.
    """

    def as_expr(self):
        from ._expressions import value

        return value(self)


@dataclasses.dataclass
class Choice(Sequence, BaseChoice):
    """A choice among an enumerated set of outcomes."""

    outcomes: list[typing.Any]
    name: typing.Optional[str] = None
    chosen_outcome_idx: typing.Optional[int] = None

    def __post_init__(self):
        if not self.outcomes:
            raise TypeError("Choice should be given at least one outcome.")
        self.outcomes = [
            _with_fields(out, in_choice=self.name) for out in self.outcomes
        ]

    def take_outcome(self, idx):
        """
        Remove an outcome from the choice.

        This does not modify the choice in-place. It returns a tuple containing
        the taken outcome and a new, reduced choice which does not contain the
        taken outcome.

        This function is used to split choices as necessary when constructing a
        hyperparameter grid.

        >>> from skrub._tuning import choose_from
        >>> choice = choose_from([1, 2, 3], name='number')
        >>> choice
        choose_from([1, 2, 3], name='number')
        >>> choice.take_outcome(1)
        (Outcome(value=2, name=None, in_choice='number'), choose_from([1, 3], name='number'))
        >>> choice
        choose_from([1, 2, 3], name='number')
        """  # noqa : E501
        out = self.outcomes[idx]
        rest = self.outcomes[:idx] + self.outcomes[idx + 1 :]
        if not rest:
            return out, None
        chosen_kwargs = {}
        if idx == self.chosen_outcome_idx:
            chosen_kwargs = {"chosen_outcome_idx": None}
        return out, _with_fields(self, outcomes=rest, **chosen_kwargs)

    def map_values(self, func):
        """
        Apply ``func`` to each of the outcomes' value.

        This does not modify the choice nor its outcomes in-place. It returns a
        new choice where each of the outcome's value has been replaced by its
        image through ``func``. The choice name and outcome names are
        preserved.

        >>> from skrub._tuning import choose_from
        >>> choice = choose_from(
        ...     {"a": "outcome a", "b": "outcome b"}, name="the choice"
        ... )
        >>> choice
        choose_from({'a': 'outcome a', 'b': 'outcome b'}, name='the choice')
        >>> choice.map_values(str.upper)
        choose_from({'a': 'OUTCOME A', 'b': 'OUTCOME B'}, name='the choice')
        >>> choice
        choose_from({'a': 'outcome a', 'b': 'outcome b'}, name='the choice')
        """
        outcomes = [_with_fields(out, value=func(out.value)) for out in self.outcomes]
        return _with_fields(self, outcomes=outcomes)

    def default(self):
        """Default outcome: the first one in the list."""
        return self.outcomes[0]

    def chosen_outcome_or_default(self):
        if self.chosen_outcome_idx is not None:
            return self.outcomes[self.chosen_outcome_idx]
        return self.default()

    def match(self, outcome_mapping):
        values = [unwrap(outcome) for outcome in self.outcomes]
        same = True
        try:
            if set(values) != set(outcome_mapping.keys()):
                same = False
        except TypeError:
            same = False
        if not same:
            raise ValueError("outcome mapping must provide a result for each outcome.")
        return Match(self, outcome_mapping)

    def __repr__(self):
        if self.outcomes[0].name is None:
            args = [out.value for out in self.outcomes]
        else:
            args = {out.name: out.value for out in self.outcomes}
        args_r = _utils.repr_args(
            (args,),
            {"name": self.name},
            defaults={"name": None},
        )
        return f"choose_from({args_r})"

    # The following methods provide the interface of a Sequence so that a
    # hyperparameter grid given to ``GridSearchCV`` can contain a ``Choice``.

    def __getitem__(self, item):
        return self.outcomes[item]

    def __len__(self):
        return len(self.outcomes)

    def __iter__(self):
        return iter(self.outcomes)


@dataclasses.dataclass
class Match:
    choice: Choice
    outcome_mapping: dict

    def match(self, outcome_mapping):
        return Match(
            self.choice,
            {k: outcome_mapping[v] for k, v in self.outcome_mapping.items()},
        )

    def as_expr(self):
        from ._expressions import value

        return value(self)


def choose_from(outcomes, name=None):
    """Construct a choice among several possible outcomes.

    Outcomes can be provided in a list:

    >>> from skrub import choose_from
    >>> choose_from([1, 2])
    choose_from([1, 2])

    They can also be provided in a dictionary to give a name to each outcome:

    >>> choose_from({'one': 1, 'two': 2})
    choose_from({'one': 1, 'two': 2})

    The choice itself can also be given a name:

    >>> choose_from({'one': 1, 'two': 2}, name='my favorite number')
    choose_from({'one': 1, 'two': 2}, name='my favorite number')
    """
    if isinstance(outcomes, typing.Mapping):
        prepared_outcomes = [Outcome(val, key) for key, val in outcomes.items()]
    else:
        prepared_outcomes = [Outcome(val) for val in outcomes]
    return Choice(prepared_outcomes, name=name)


def unwrap(obj):
    """Extract the value from an Outcome or a plain value.

    If the input is a plain value, it is returned unchanged.

    >>> from skrub._tuning import choose_from, unwrap
    >>> choice = choose_from([1, 2])
    >>> outcome = choice.default()
    >>> outcome
    Outcome(value=1, name=None, in_choice=None)
    >>> unwrap(outcome)
    1
    >>> unwrap(1)
    1
    """
    if isinstance(obj, Outcome):
        return obj.value
    return obj


def unwrap_default(obj):
    """Extract a value from a Choice, Outcome or plain value.

    If the input is a Choice, the default outcome is used.
    For other inputs, behaves like ``unwrap``.

    >>> from skrub._tuning import choose_from, unwrap_default
    >>> choice = choose_from([1, 2])
    >>> choice
    choose_from([1, 2])
    >>> choice.default()
    Outcome(value=1, name=None, in_choice=None)
    >>> unwrap_default(choice)
    1
    >>> unwrap_default(choice.default())
    1
    >>> unwrap_default(1)
    1
    """
    if isinstance(obj, Match):
        return obj.outcome_mapping[unwrap_default(obj.choice)]
    if isinstance(obj, BaseChoice):
        return obj.default().value
    return unwrap(obj)


def unwrap_chosen_or_default(obj):
    if isinstance(obj, Match):
        return obj.outcome_mapping[unwrap_chosen_or_default(obj.choice)]
    if isinstance(obj, BaseChoice):
        return obj.chosen_outcome_or_default().value
    return unwrap(obj)


class Optional(Choice):
    """A choice between something and nothing."""

    def __repr__(self):
        args = _utils.repr_args(
            (unwrap_default(self),), {"name": self.name}, defaults={"name": None}
        )
        return f"optional({args})"


def optional(value, name=None):
    """Construct a choice between a value and ``None``.

    This is useful for optional steps in a pipeline. If we want to try our
    pipeline with or without dimensionality reduction, we can add a step such
    as:

    >>> from sklearn.decomposition import PCA
    >>> from skrub import optional
    >>> optional(PCA(), name="use dim reduction")
    optional(PCA(), name='use dim reduction')

    The constructed parameter grid will include a version of the pipeline with
    the PCA and one without.
    """
    return Optional([Outcome(value, "true"), Outcome(None, "false")], name=name)


class BoolChoice(Choice):
    def __repr__(self):
        return "choose_bool()"

    def if_else(self, if_true, if_false):
        return self.match({True: if_true, False: if_false})


def choose_bool(name=None):
    """Construct a choice between False and True."""
    return BoolChoice([Outcome(True), Outcome(False)], name=name)


def _check_bounds(low, high, log):
    if high < low:
        raise ValueError(
            f"'high' must be greater than 'low', got low={low}, high={high}"
        )
    if log and low <= 0:
        raise ValueError(f"To use log space 'low' must be > 0, got low={low}")


@dataclasses.dataclass
class NumericOutcome(Outcome):
    """An outcome in a choice from a numeric range.

    In addition to the attributes of ``Outcome``, this records whether the
    choice is made on a log scale (``is_from_log_scale``). This is useful to
    choose the scale of the plot axis on which those values are displayed for
    example in the parallel coordinate plots.
    """

    value: typing.Union[int, float]
    is_from_log_scale: bool = False
    name: typing.Optional[str] = None
    in_choice: typing.Optional[str] = None


def _repr_numeric_choice(choice):
    args = _utils.repr_args(
        (choice.low, choice.high),
        {
            "log": choice.log,
            "n_steps": getattr(choice, "n_steps", None),
            "name": choice.name,
        },
        defaults={"log": False, "n_steps": None, "name": None},
    )
    if choice.to_int:
        return f"choose_int({args})"
    return f"choose_float({args})"


class BaseNumericChoice(BaseChoice):
    """Base class to help identify numeric choices with ``isinstance``.

    It also provides a helper to wrap a (randomly sampled) number in a
    ``NumericOutcome``.
    """

    def _wrap_outcome(self, value):
        return NumericOutcome(value, is_from_log_scale=self.log, in_choice=self.name)

    def chosen_outcome_or_default(self):
        if self.chosen_outcome is not None:
            return self.chosen_outcome
        return self.default()


@dataclasses.dataclass
class NumericChoice(BaseNumericChoice):
    """A choice within a numeric range."""

    low: float
    high: float
    log: bool
    to_int: bool
    name: str = None
    chosen_outcome: typing.Optional[typing.Union[int, float]] = None

    def __post_init__(self):
        _check_bounds(self.low, self.high, self.log)
        if self.log:
            self._distrib = stats.loguniform(self.low, self.high)
        else:
            self._distrib = stats.uniform(self.low, self.high)

    def rvs(self, size=None, random_state=None):
        value = self._distrib.rvs(size=size, random_state=random_state)
        if self.to_int:
            value = value.astype(int)
        if size is None:
            return self._wrap_outcome(value)
        return [self._wrap_outcome(v) for v in value]

    def default(self):
        low, high = self.low, self.high
        if self.log:
            low, high = np.log(low), np.log(high)
        midpoint = np.mean([low, high])
        if self.log:
            midpoint = np.exp(midpoint)
        if self.to_int:
            midpoint = np.round(midpoint).astype(int)
        return self._wrap_outcome(midpoint)

    def __repr__(self):
        return _repr_numeric_choice(self)


@dataclasses.dataclass
class DiscretizedNumericChoice(BaseNumericChoice, Sequence):
    """A choice from a numeric range discretized by providing ``n_steps``."""

    low: float
    high: float
    n_steps: int
    log: bool
    to_int: bool
    name: str = None
    chosen_outcome: typing.Optional[typing.Union[int, float]] = None

    def __post_init__(self):
        _check_bounds(self.low, self.high, self.log)
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got: {self.n_steps}")
        if self.log:
            low, high = np.log(self.low), np.log(self.high)
        else:
            low, high = self.low, self.high
        self.grid = np.linspace(low, high, self.n_steps)
        if self.log:
            self.grid = np.exp(self.grid)
        if self.to_int:
            self.grid = np.round(self.grid).astype(int)

    def rvs(self, size=None, random_state=None):
        random_state = check_random_state(random_state)
        value = random_state.choice(self.grid, size=size)
        if size is None:
            return self._wrap_outcome(value)
        return [self._wrap_outcome(v) for v in value]

    def default(self):
        value = self.grid[(len(self.grid) - 1) // 2]
        return self._wrap_outcome(value)

    def __repr__(self):
        return _repr_numeric_choice(self)

    # Provide the Sequence interface so that discretized numeric choices are
    # compatible with ``GridSearchCV`` (in addition to ``RandomizedSearchCV``).

    def __getitem__(self, item):
        return self.grid[item]

    def __len__(self):
        return len(self.grid)

    def __iter__(self):
        return iter(self.grid)


def choose_float(low, high, log=False, n_steps=None, name=None):
    """Construct a choice of floating-point numbers from a numeric range."""
    if n_steps is None:
        return NumericChoice(low, high, log=log, to_int=False, name=name)
    return DiscretizedNumericChoice(
        low, high, log=log, to_int=False, n_steps=n_steps, name=name
    )


def choose_int(low, high, log=False, n_steps=None, name=None):
    """Construct a choice of integers from a numeric range."""
    if n_steps is None:
        return NumericChoice(low, high, log=log, to_int=True, name=name)
    return DiscretizedNumericChoice(
        low, high, log=log, to_int=True, n_steps=n_steps, name=name
    )


#
# ``expand_grid``
# ===============
#
# Now that we have the representations for different kinds of choices and
# functions to construct them, the rest of this module provides the
# implementation of ``expand_grid``, which turns estimators containing choices
# into a grid of hyperparameters that can be understood by ``GridSearchCV`` and
# ``RandomizedSearchCV``.
#


@dataclasses.dataclass
class PlaceHolder:
    """Inserted in place of parameters that will get replaced with set_params anyway.

    This allows controlling the repr and clarifying what will take the place of
    a given parameter.

    For example the ``PlaceHolder`` repr is ``<α>`` in this grid:

    >>> from skrub._tuning import expand_grid, grid_description, choose_from
    >>> from sklearn.linear_model import Ridge, Lasso
    >>> alpha = choose_from([10, 100], name="α")
    >>> grid = {'regressor': choose_from([Lasso(alpha=alpha), Ridge(alpha=alpha)])}
    >>> print(grid_description(expand_grid(grid)))
    - 'regressor': Lasso(alpha=<α>)
      'α':
          - 10
          - 100
    - 'regressor': Ridge(alpha=<α>)
      'α':
          - 10
          - 100
    """

    name: typing.Optional[str] = None

    def __repr__(self):
        if self.name is not None:
            return f"<{self.name}>"
        return "..."


def _find_param_choices(obj):
    """Find all the choices in an estimator.

    This uses scikit-learn's ``get_params`` to find all hyperparameters
    (attributes) that are ``BaseChoice``s in the input estimator (or its nested
    sub-estimators).

    >>> from skrub._tuning import _find_param_choices, choose_float, choose_int
    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.linear_model import Ridge

    >>> reg = BaggingRegressor(
    ...     Ridge(alpha=choose_float(1.0, 10.0)), n_estimators=choose_int(10, 30)
    ... )
    >>> _find_param_choices(reg)
    {'estimator__alpha': choose_float(1.0, 10.0), 'n_estimators': choose_int(10, 30)}
    """  # noqa: E501
    if not hasattr(obj, "get_params"):
        return []
    params = obj.get_params(deep=True)
    return {k: v for k, v in params.items() if isinstance(v, BaseChoice)}


def contains_choice(estimator):
    """Return ``True`` if an estimator is a choice or contains one."""
    return isinstance(estimator, Choice) or bool(_find_param_choices(estimator))


def with_default_params(estimator):
    """Replace all choices by their default value in an estimator's parameters.

    If the estimator itself is a choice, also resolve that choice to its
    default value.

    The input itself is not modified, a new object is returned.

    >>> from skrub._tuning import choose_float, choose_int, choose_from
    >>> from skrub._tuning import with_default_params

    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.linear_model import Ridge

    >>> ridge = Ridge(alpha=choose_float(1.0, 100.0, log=True))
    >>> bag = BaggingRegressor(ridge, n_estimators=choose_int(10, 20))
    >>> reg = choose_from([ridge, bag])
    >>> reg
    choose_from([Ridge(alpha=choose_float(1.0, 100.0, log=True)), BaggingRegressor(estimator=Ridge(alpha=choose_float(1.0, 100.0, log=True)),
                     n_estimators=choose_int(10, 20))])
    >>> with_default_params(reg) # doctest: +SKIP
    Ridge(alpha=10.000000000000002)
    >>> reg
    choose_from([Ridge(alpha=choose_float(1.0, 100.0, log=True)), BaggingRegressor(estimator=Ridge(alpha=choose_float(1.0, 100.0, log=True)),
                     n_estimators=choose_int(10, 20))])
    >>>
    """  # noqa: E501
    estimator = unwrap_default(estimator)
    if not hasattr(estimator, "set_params"):
        return estimator
    estimator = clone(estimator)
    while param_choices := _find_param_choices(estimator):
        params = {k: unwrap_default(v) for k, v in param_choices.items()}
        estimator.set_params(**params)
    return estimator


def expand_grid(grid):
    grid = _split_grid(grid)
    new_grid = []
    for subgrid in grid:
        new_subgrid = {}
        for k, v in subgrid.items():
            if isinstance(v, Outcome):
                v = Choice([v], name=v.in_choice)
            new_subgrid[k] = v
        new_grid.append(new_subgrid)
        _check_name_collisions(new_subgrid)
    return new_grid


"""
Constructing the grid of hyperparameters is done by repeating 2 steps.

``_extract_choices`` looks at all estimators at the top level of the grid, i.e.
estimators that are not part of a Choice. If one of them contains a choice as
one of its hyperparameters, the choice is extracted from the estimator and
placed onto the grid.

For example:

>>> _extract_choices({'regressor': Ridge(alpha=choose_from([1., 10.]))}) # doctest: +SKIP
{'regressor__alpha': choose_from([1.0, 10.0])}

``alpha`` has been moved out of the ``Ridge`` and placed onto the grid. Once
this is done, as there are no alternative to the ``Ridge`` in this example, it
is not needed anymore and is removed from the grid to reduce clutter. It could
have been retained in which case we would also have ``"regressor": [Ridge()]``
but that would be redundant as there is only one option in the ``[Ridge()]``
list, which is already the value set on the pipeline.

When the estimator itself is a choice, and the outcomes contain hyperparameter
choices, we cannot mix all of their hyperparameter ranges in the same
dictionary. In this case we need to split the grid into subgrids, one for each
of the possible estimators. This is handled by ``_split_grid``.

>>> split = _split_grid({'dim_reduction': choose_from([PCA(n_components=n), SelectKBest(k=n)])}) # doctest: +SKIP
>>> pprint(split)
[{'dim_reduction': Outcome(value=PCA(n_components=<dim_reduction__n_components>),
                           name=None,
                           in_choice=None),
  'dim_reduction__n_components': choose_from([10, 20])},
 {'dim_reduction': Outcome(value=SelectKBest(k=<dim_reduction__k>),
                           name=None,
                           in_choice=None),
  'dim_reduction__k': choose_from([10, 20])}]

We see that the grid has been split into one subgrid for the case
'dim_reduction=PCA' and one for the case 'dim_reduction=SelectKBest'. Then,
``_split_grid`` calls itself recursively on each subgrid. As ``_split_grid``
starts by applying ``_extract_choices``, the ``dim_reduction__k`` and
``dim_reduction__n_components`` have been extracted from the estimators and
placed on their subgrids, and the result is now ready to be given to
``GridSearchCV`` or ``RandomizedSearchCV`` after some small post-processing
performed by ``expand_grid``.
"""  # noqa: E501


def _extract_choices(grid):
    """Extract hyperparameter ranges and place them on the grid.

    This only considers estimators that are not inside of a Choice.
    Any of their hyperparameters that is a Choice is extracted and placed on
    the grid.
    """
    new_grid = {}
    for param_name, param in grid.items():
        if isinstance(param, Choice) and len(param.outcomes) == 1:
            param = param.outcomes[0]
        if isinstance(param, (Outcome, BaseChoice)):
            new_grid[param_name] = param
        else:
            # In this case we have a 'raw' estimator that has not been wrapped
            # in an Outcome. Therefore it is not part of a choice itself, but it
            # contains a choice. We will pull out the choices to include them in the
            # grid, but the estimator itself does not need to be in the grid so we
            # don't include it to keep the grid more compact.
            param = Outcome(param)
        if isinstance(param, BaseChoice):
            # If the grid item is a Choice, we leave it alone as it requires a
            # split, it will be handled in ``split_grid``.
            continue

        # Extract any choices contained in the estimator and put them on the
        # grid.
        all_subparam_choices = _find_param_choices(param.value)
        if not all_subparam_choices:
            continue
        placeholders = {}
        for subparam_name, subparam_choice in all_subparam_choices.items():
            subparam_id = f"{param_name}__{subparam_name}"
            placeholder_name = subparam_id if (n := subparam_choice.name) is None else n
            placeholders[subparam_name] = PlaceHolder(placeholder_name)
            new_grid[subparam_id] = subparam_choice
        if param_name in new_grid:
            estimator = clone(param.value)
            estimator.set_params(**placeholders)
            new_grid[param_name] = _with_fields(param, value=estimator)
    return new_grid


def _split_grid(grid):
    """Construct the hyperparameter grid.

    It starts by calling ``_extract_choices`` to move hyperparameter choices
    out of the estimators and placing them on the grid.

    Then, whenever a Choice contains an estimator that itself has a
    hyperparameter grid, this estimator needs to be expanded in its own
    subgrid.

    In this case ``_split_grid`` creates the subgrids, then calls itself
    recursively on each of them.
    """
    grid = _extract_choices(grid)
    for param_name, param in grid.items():
        if not isinstance(param, Choice):
            continue
        for idx, outcome in enumerate(param.outcomes):
            if _find_param_choices(outcome.value):
                # This is an outcome in a choice, and it contains choices
                # itself. For example the SelectKBest in {'dim_reduc':
                # choose_from([SelectKBest(k=choose_from([10, 20])), PCA()])}
                # -- the grid needs to be split so that the k can be extracted
                # into a separate subgrid.
                grid_1 = grid.copy()
                grid_1[param_name] = outcome
                _, rest = param.take_outcome(idx)
                if rest is None:
                    return _split_grid(grid_1)
                grid_2 = grid.copy()
                grid_2[param_name] = rest
                return [*_split_grid(grid_1), *_split_grid(grid_2)]
    return [grid]


def _check_name_collisions(subgrid):
    """
    When we need to evaluate combinations of 2 parameters, they cannot be given the
    same ``name``. For example:

    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.decomposition import PCA
    >>> from skrub._tuning import expand_grid, choose_from

    >>> grid = {
    ...     "kbest": SelectKBest(k=choose_from([10, 20], name="my param")),
    ...     "ridge": Ridge(alpha=choose_from([10, 20], name="my param")),
    ... }

    Here we will need to evaluate 4 combinations of the value for ``k`` and for
    ``alpha``. However we have given them the same alias, "my param". When looking
    at hyperparameter search results we would not know to which of those "my param"
    refers. So it is an error:

    >>> expand_grid(grid)
    Traceback (most recent call last):
        ...
    ValueError: Parameter alias 'my param' used for several parameters: (('kbest__k', choose_from([10, 20], name='my param')), ('ridge__alpha', choose_from([10, 20], name='my param'))).

    Note that reusing the same name for estimators that are different outcomes in a
    choice is fine. For example in this grid:

    >>> grid = {
    ...     "reduce_dim": choose_from(
    ...         [
    ...             SelectKBest(k=choose_from([10, 20], name="my param")),
    ...             PCA(n_components=choose_from([30, 40], name="my param")),
    ...         ]
    ...     )
    ... }

    For each run we are using _either_ the ``SelectKBest`` _or_ ``PCA``. This is
    different from the example above where each run used _both_ the ``SelectKBest``
    _and_ the ``Ridge``. Therefore in this case there is no ambiguity: if we are in
    the case where ``reduce_dim`` is ``SelectKBest``, we know that "my param" refers
    to ``k``. Conversely if ``reduce_dim`` is ``PCA`` we know that "my param" refers
    to ``n_components``. So we can expand the grid without errors:

    >>> expand_grid(grid)
    [{'reduce_dim': choose_from([SelectKBest(k=<my param>)]), 'reduce_dim__k': choose_from([10, 20], name='my param')}, {'reduce_dim': choose_from([PCA(n_components=<my param>)]), 'reduce_dim__n_components': choose_from([30, 40], name='my param')}]
    """  # noqa: E501
    all_names = {}
    for param_id, param in subgrid.items():
        name = param.name or param_id
        if name in all_names:
            raise ValueError(
                f"Parameter alias {name!r} used for "
                f"several parameters: {all_names[name], (param_id, param)}."
            )
        all_names[name] = (param_id, param)


#
# A few helpers to display parameter grids or nodes in a parameter grid
#


def write_indented(prefix, text, ostream):
    istream = io.StringIO(text)
    ostream.write(prefix)
    ostream.write(next(istream))
    for line in istream:
        ostream.write(" " * len(prefix))
        ostream.write(line)
    return ostream.getvalue()


def grid_description(grid):
    buf = io.StringIO()
    for subgrid in grid:
        prefix = "- "
        for k, v in subgrid.items():
            if v.name is not None:
                k = v.name
            if isinstance(v, BaseNumericChoice):
                # no need to repeat the name (already in the key) hence name(None)
                write_indented(
                    f"{prefix}{k!r}: ", f"{_with_fields(v, name=None)}\n", buf
                )
            elif len(v.outcomes) == 1:
                write_indented(f"{prefix}{k!r}: ", f"{v.outcomes[0]}\n", buf)
            else:
                buf.write(f"{prefix}{k!r}:\n")
                for outcome in v.outcomes:
                    write_indented("      - ", f"{outcome}\n", buf)
            prefix = "  "
    return buf.getvalue()


def params_description(grid_entry):
    buf = io.StringIO()
    for param_id, param in grid_entry.items():
        choice_name = param.in_choice or param_id
        value = param.name or param.value
        write_indented(f"{choice_name!r}: ", f"{value!r}\n", buf)
    return buf.getvalue()
