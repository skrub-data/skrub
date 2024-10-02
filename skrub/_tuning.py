"""
Choices and Outcomes
--------------------

This module provides classes to represent a range of hyperparameters for an
estimator, or a range of different estimators.

The main reason for those custom classes is that we can use them directly to
initialize scikit-learn estimators or pipeline steps. Then, because of their
special type, the ``Recipe`` can easily spot them and handle them appropriately
to construct a grid of hyperparameters.

Some hyperparameters take their value from a discrete set (for example the
``svd_solver`` of a ``PCA`` can be "auto", "full", "covariance_eigh", ...). Some
others take their values from a range of real or integral numbers (for example
the ``n_components`` of a ``PCA`` is an int between 0 and the smallest dimension
of ``X``). This module allows to represent all those kinds of ranges.

A "choice" represents a range of things from which we can choose. Each of those
things --the result of choosing-- is referred to here as an "outcome".

Imagine that we want a dimensionality reduction step in our machine-learning
model, and that we want to try both a PCA or feature selection:

>>> from sklearn.decomposition import PCA
>>> from sklearn.feature_selection import SelectKBest

We can represent this range of possibilities with a Choice. The ``choose_from``
factory constructs it for us.

>>> from skrub._tuning import choose_from
>>> dim_reduction = choose_from([PCA(), SelectKBest()])
>>> dim_reduction
choose_from([PCA(), SelectKBest()])
>>> type(dim_reduction)
<class 'skrub._tuning.Choice'>

>>> dim_reduction.outcomes
[Outcome(value=PCA(), name=None, in_choice=None), Outcome(value=SelectKBest(), name=None, in_choice=None)]

(Ignore ``name`` and ``in_choice`` which refer to optional human-readable labels that we
have not provided here, they are discussed later.)

Choices provide the sequence interface which is used by ``GridSearchCV`` to
index the possible outcomes:
>>> list(dim_reduction)
[Outcome(value=PCA(), name=None, in_choice=None), Outcome(value=SelectKBest(), name=None, in_choice=None)]

Note that it is important that the list is wrapped in a special class. When a
Choice is used as a parameter of an estimator, we can recognize it by its type
and extract it to build the hyperparameter grid. If we used a plain list
instead, we would have no way to know if it represents a choice or simply a
parameter value. For example the ``alphas`` parameter of ``RidgeCV`` expects a
list of numbers. Moreover the ``Choice`` adds some features that are useful for
inspecting hyperparameter search results as shown below.

It is possible to give the choice a human-readable ``name``, which is used for
example by the ``Recipe`` to refer to it when showing hyperparameter search
results. This can provide a more readable display in the results table or in the
parallel coordinate plots.

>>> choose_from([PCA(), SelectKBest()], name='dim reduction')
choose_from([PCA(), SelectKBest()], name='dim reduction')

Here is another example where giving a name can be useful (note that choices
can be nested arbitrarily):

>>> n_dims = choose_from([10, 20, 30], name='n dimensions')
>>> choose_from([PCA(n_components=n_dims), SelectKBest(k=n_dims)])
choose_from([PCA(n_components=choose_from([10, 20, 30], name='n dimensions')), SelectKBest(k=choose_from([10, 20, 30], name='n dimensions'))])

We are applying the same label 'n dimensions' to ``k`` and ``n_components``
because they play the same role in our pipeline.

Moreover, each of the outcomes can also be given a name. This is acheived by
using a dictionary rather than a list:

>>> dim_reduction = choose_from({'pca': PCA(), 'k best': 'SelectKBest'}, name='dim reduction')
>>> dim_reduction
choose_from({'pca': PCA(), 'k best': 'SelectKBest'}, name='dim reduction')

Each of the outcomes records its name so that when they are passed to an
estimator and recorded in grid search results they can be inspected to know
their name and which choice they came from.

>>> dim_reduction.outcomes
[Outcome(value=PCA(), name='pca', in_choice='dim reduction'), Outcome(value='SelectKBest', name='k best', in_choice='dim reduction')]

Again, this is a way to give a human-readable label to each outcome rather than
relying on the value's ``__repr__()`` which can be verbose or uninformative.

A choice between an enumerated set of values is represented by a ``Choice`` and
constructed with ``choose_from`` as shown above.
Additional classes are provided to represent ranges of numeric values.

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

If we need floating-point numbers rather than ints we use ``choose_float``:
>>> alpha = choose_float(.01, 100, log=True, name='α')
>>> alpha
choose_float(0.01, 100, log=True, name='α')
>>> alpha.rvs() # doctest: +SKIP
NumericOutcome(value=16.656593316727974, name=None, in_choice='α', is_from_log_scale=True)

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
>>> list(n_dims)
[10, 18, 32, 56, 100]

Finally, it is common to choose between something or nothing. The last type of
choice is a shorthand for a choice between one value and ``None``:

>>> from skrub._tuning import optional
>>> feature_selection = optional(SelectKBest(), name='feature selection')
>>> feature_selection
optional(SelectKBest(), name='feature selection')
>>> type(feature_selection)
<class 'skrub._tuning.Optional'>
>>> list(feature_selection)
[Outcome(value=SelectKBest(), name='true', in_choice='feature selection'), Outcome(value=None, name='false', in_choice='feature selection')]

Constructing a hyperparameter grid
----------------------------------

TODO

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


def _with_fields(obj, **fields):
    return obj.__class__(
        **({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)} | fields)
    )


@dataclasses.dataclass
class Outcome:
    value: typing.Any
    name: typing.Optional[str] = None
    in_choice: typing.Optional[str] = None

    def __str__(self):
        if self.name is not None:
            return repr(self.name)
        return repr(self.value)


class BaseChoice:
    pass


@dataclasses.dataclass
class Choice(Sequence, BaseChoice):
    outcomes: list[typing.Any]
    name: typing.Optional[str] = None

    def __post_init__(self):
        if not self.outcomes:
            raise TypeError("Choice should be given at least one outcome.")
        self.outcomes = [
            _with_fields(out, in_choice=self.name) for out in self.outcomes
        ]

    def take_outcome(self, idx):
        out = self.outcomes[idx]
        rest = self.outcomes[:idx] + self.outcomes[idx + 1 :]
        if not rest:
            return out, None
        return out, _with_fields(self, outcomes=rest)

    def map_values(self, func):
        outcomes = [_with_fields(out, value=func(out.value)) for out in self.outcomes]
        return _with_fields(self, outcomes=outcomes)

    def default(self):
        return self.outcomes[0]

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

    def __getitem__(self, item):
        return self.outcomes[item]

    def __len__(self):
        return len(self.outcomes)

    def __iter__(self):
        return iter(self.outcomes)


def choose_from(outcomes, name=None):
    if isinstance(outcomes, typing.Mapping):
        prepared_outcomes = [Outcome(val, key) for key, val in outcomes.items()]
    else:
        prepared_outcomes = [Outcome(val) for val in outcomes]
    return Choice(prepared_outcomes, name=name)


class Optional(Choice):
    def __repr__(self):
        args = _utils.repr_args(
            (unwrap_default(self),), {"name": self.name}, defaults={"name": None}
        )
        return f"optional({args})"


def optional(value, name=None):
    return Optional([Outcome(value, "true"), Outcome(None, "false")], name=name)


def _check_bounds(low, high, log):
    if high < low:
        raise ValueError(
            f"'high' must be greater than 'low', got low={low}, high={high}"
        )
    if log and low <= 0:
        raise ValueError(f"To use log space 'low' must be > 0, got low={low}")


@dataclasses.dataclass
class NumericOutcome(Outcome):
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
    pass


@dataclasses.dataclass
class NumericChoice(BaseNumericChoice):
    low: float
    high: float
    log: bool
    to_int: bool
    name: str = None

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
        return NumericOutcome(value, is_from_log_scale=self.log, in_choice=self.name)

    def default(self):
        low, high = self.low, self.high
        if self.log:
            low, high = np.log(low), np.log(high)
        midpoint = np.mean([low, high])
        if self.log:
            midpoint = np.exp(midpoint)
        if self.to_int:
            midpoint = np.round(midpoint).astype(int)
        return NumericOutcome(midpoint, is_from_log_scale=self.log, in_choice=self.name)

    def __repr__(self):
        return _repr_numeric_choice(self)


@dataclasses.dataclass
class DiscretizedNumericChoice(BaseNumericChoice, Sequence):
    low: float
    high: float
    n_steps: int
    log: bool
    to_int: bool
    name: str = None

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
        return NumericOutcome(value, is_from_log_scale=self.log, in_choice=self.name)

    def default(self):
        value = self.grid[(len(self.grid) - 1) // 2]
        return NumericOutcome(value, is_from_log_scale=self.log, in_choice=self.name)

    def __getitem__(self, item):
        return self.grid[item]

    def __len__(self):
        return len(self.grid)

    def __iter__(self):
        return iter(self.grid)

    def __repr__(self):
        return _repr_numeric_choice(self)


def choose_float(low, high, log=False, n_steps=None, name=None):
    if n_steps is None:
        return NumericChoice(low, high, log=log, to_int=False, name=name)
    return DiscretizedNumericChoice(
        low, high, log=log, to_int=False, n_steps=n_steps, name=name
    )


def choose_int(low, high, log=False, n_steps=None, name=None):
    if n_steps is None:
        return NumericChoice(low, high, log=log, to_int=True, name=name)
    return DiscretizedNumericChoice(
        low, high, log=log, to_int=True, n_steps=n_steps, name=name
    )


@dataclasses.dataclass
class Placeholder:
    name: typing.Optional[str] = None

    def __repr__(self):
        if self.name is not None:
            return f"<{self.name}>"
        return "..."


def unwrap_default(obj):
    if isinstance(obj, BaseChoice):
        return obj.default().value
    if isinstance(obj, Outcome):
        return obj.value
    return obj


def unwrap(obj):
    if isinstance(obj, Outcome):
        return obj.value
    return obj


def contains_choice(estimator):
    return isinstance(estimator, Choice) or bool(_find_param_choices(estimator))


def set_params_to_default(estimator):
    estimator = unwrap_default(estimator)
    if not hasattr(estimator, "set_params"):
        return estimator
    estimator = clone(estimator)
    while param_choices := _find_param_choices(estimator):
        params = {k: unwrap_default(v) for k, v in param_choices.items()}
        estimator.set_params(**params)
    return estimator


def _find_param_choices(obj):
    if not hasattr(obj, "get_params"):
        return []
    params = obj.get_params(deep=True)
    return {k: v for k, v in params.items() if isinstance(v, BaseChoice)}


def _extract_choices(grid):
    new_grid = {}
    for param_name, param in grid.items():
        if isinstance(param, Choice) and len(param.outcomes) == 1:
            param = param.outcomes[0]
        if isinstance(param, (Outcome, BaseChoice)):
            new_grid[param_name] = param
        else:
            # In this case we have a 'raw' estimator that has not been wrapped
            # in an Outcome. Therefore it is not part of a choice itself, but it
            # contains a choice. We pull out the choices to include them in the
            # grid, but the param itself does not need to be in the grid so we
            # don't include it to keep the grid more compact.
            param = Outcome(param)
        if isinstance(param, BaseChoice):
            continue
        all_subparam_choices = _find_param_choices(param.value)
        if not all_subparam_choices:
            continue
        placeholders = {}
        for subparam_name, subparam_choice in all_subparam_choices.items():
            subparam_id = f"{param_name}__{subparam_name}"
            placeholder_name = subparam_id if (n := subparam_choice.name) is None else n
            placeholders[subparam_name] = Placeholder(placeholder_name)
            new_grid[subparam_id] = subparam_choice
        if param_name in new_grid:
            estimator = clone(param.value)
            estimator.set_params(**placeholders)
            new_grid[param_name] = _with_fields(param, value=estimator)
    return new_grid


def _split_grid(grid):
    grid = _extract_choices(grid)
    for param_name, param in grid.items():
        if not isinstance(param, Choice):
            continue
        for idx, outcome in enumerate(param.outcomes):
            if _find_param_choices(outcome.value):
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
    all_names = {}
    for param_id, param in subgrid.items():
        name = param.name or param_id
        if name in all_names:
            raise ValueError(
                f"Parameter alias {name!r} used for "
                f"several parameters: {all_names[name], (param_id, param)}."
            )
        all_names[name] = (param_id, param)


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
