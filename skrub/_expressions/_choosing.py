import dataclasses
import functools
import typing
from collections.abc import Sequence

import numpy as np
from scipy import stats
from sklearn.utils import check_random_state

from .. import _utils
from ._utils import NULL


def _with_fields(obj, **fields):
    """
    Make a copy of a dataclass instance with different values for some of the attributes
    """
    return obj.__class__(
        **({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)} | fields)
    )


def _wrap_getitem(getitem):
    @functools.wraps(getitem)
    def __getitem__(self, key):
        if isinstance(key, str):
            if key == self.keys()[0]:
                return self
            raise KeyError(key)
        return getitem(self, key)

    return __getitem__


class BaseChoice:
    """Base class for all choices (enumerated or numeric range)."""

    __hash__ = None

    def as_expr(self):
        """Wrap the choice in an expression.

        ``choice.as_expr()`` is a convenience shorthand for ``skrub.as_expr(choice)``.
        """
        from ._expressions import as_expr

        return as_expr(self)

    # We provide the interface that enables dict unpacking with `**choice`:
    # `keys()` and `__getitem__`. This is to offer syntactic sugar to avoid
    # repeating the choice name when it is the same as a parameter name (a
    # common case when the choice is an estimator's parameter).
    #
    # Discretized numeric choices are also sequences so that they can be stuck
    # directly into a GridSearchCV's param grid (as well as
    # RandomizedSearchCV's). To avoid interfering with that, if the subclass
    # already defines a __getitem__ we wrap it to add handling of the special
    # case where the key is a str (the parameter name), and don't modify its
    # behavior when the key is an int.

    def keys(self):
        return [self.name.rsplit("__", maxsplit=1)[-1]]

    def __init_subclass__(cls):
        if (cls_getitem := cls.__dict__.get("__getitem__", None)) is not None:
            setattr(cls, "__getitem__", _wrap_getitem(cls_getitem))

    @_wrap_getitem
    def __getitem__(self, key):
        raise TypeError(f"key must be {self.keys()[0]!r}")


def _check_match_keys(outcome_values, mapping_keys, has_default):
    try:
        extra_keys = set(mapping_keys).difference(outcome_values)
        extra_outcomes = set(outcome_values).difference(mapping_keys)
    except TypeError as e:
        raise TypeError(f"To use `match()`, all choice outcomes must be hashable. {e}")
    if extra_keys:
        raise ValueError(
            "The following keys were found in the mapping provided to `match()` but"
            f" are not possible choice outcomes: {extra_keys!r}"
        )
    if has_default:
        return
    if extra_outcomes:
        raise ValueError(
            "The following outcomes do not have a corresponding key in the mapping"
            f" provided to `match()`: {extra_outcomes!r}. Please provide an entry for"
            " each possible outcome, or a default."
        )


def _check_name(name):
    if not isinstance(name, str):
        raise TypeError(
            f"choice `name` must be a `str`, got object of type: {type(name)}"
        )


@dataclasses.dataclass
class Choice(BaseChoice):
    """A choice among an enumerated set of outcomes."""

    outcomes: list[typing.Any]
    outcome_names: typing.Optional[list[str]]
    name: str
    chosen_outcome_idx: typing.Optional[int] = None

    def __post_init__(self):
        _check_name(self.name)
        if not self.outcomes:
            raise ValueError("Choice should be given at least one outcome.")
        if self.outcome_names is None:
            return
        if len(self.outcome_names) != len(self.outcomes):
            raise ValueError("lengths of `outcome_names` and `outcomes` do not match")
        for name in self.outcome_names:
            if not isinstance(name, str):
                raise TypeError(
                    "Outcome names (keys in the dict passed to `choose_from`) "
                    f"should be of type `str`, got: {type(name)}"
                )
        if len(set(self.outcome_names)) != len(self.outcome_names):
            raise ValueError("Outcome names should be unique")

    def map_values(self, func):
        """
        Apply ``func`` to each of the outcomes' value.

        This does not modify the choice nor its outcomes in-place. It returns a
        new choice where each of the outcome's value has been replaced by its
        image through ``func``. The choice name and outcome names are
        preserved.

        >>> from skrub import choose_from
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
        outcomes = [func(out) for out in self.outcomes]
        return _with_fields(self, outcomes=outcomes)

    def default(self):
        """Default outcome: the first one in the list."""
        return self.outcomes[0]

    def chosen_outcome_or_default(self):
        if self.chosen_outcome_idx is not None:
            return self.outcomes[self.chosen_outcome_idx]
        return self.default()

    def match(self, outcome_mapping, default=NULL):
        """Select a value depending on the outcome of the choice.

        This allows inserting several inter-dependent hyper-parameters in a
        pipeline, which all depend on the outcome of the same choice. See the
        examples below.

        Parameters
        ----------
        outcome_mapping : dict
            Maps possible outcome to the desired result. The keys must all be
            one of the possible outcomes for the choice. If no ``default`` is
            provided, there must be an entry in ``outcome_mapping`` for each
            possible choice outcome.

        default : object, optional
            The value to use for outcomes not found in ``outcome_mapping``.

        Returns
        -------
        Match
            An object which evaluates to the image of the choice through
            ``outcome_mapping``.

        Examples
        --------
        Suppose we want to encode strings differently depending on the
        supervised estimator we use. the ``MinHashEncoder`` can be a good
        choice when the downstream learner is a tree-based model, but not when
        it is a linear model. So we have 2 choices in our pipeline, the
        encoding and the learner, but they are not independent: not all
        combinations make sense.

        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import HistGradientBoostingClassifier
        >>> from sklearn.feature_selection import SelectKBest
        >>> import skrub

        >>> learner_kind = skrub.choose_from(["logistic", "hgb"], name="learner")
        >>> encoder = learner_kind.match(
        ...     {
        ...         "logistic": skrub.StringEncoder(
        ...             **skrub.choose_int(5, 20, name="string__n_components")
        ...         ),
        ...         "hgb": skrub.MinHashEncoder(
        ...             **skrub.choose_int(10, 30, name="minhash__n_components")
        ...         ),
        ...     }
        ... )
        >>> vectorizer = skrub.TableVectorizer(high_cardinality=encoder)
        >>> predictor = learner_kind.match(
        ...     {
        ...         "logistic": LogisticRegression(
        ...             **skrub.choose_float(0.01, 10.0, log=True, name="C")
        ...         ),
        ...         "hgb": HistGradientBoostingClassifier(
        ...             **skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
        ...         ),
        ...     }
        ... )
        >>> selector = SelectKBest(**skrub.choose_int(15, 25, name='k'))
        >>> X, y = skrub.X(), skrub.y()
        >>> pred = (
        ...     X.skb.apply(vectorizer)
        ...     .skb.apply(selector, y=y)
        ...     .skb.apply(predictor, y=y)
        ... )

        >>> print(pred.skb.describe_param_grid())
        - k: choose_int(15, 25, name='k')
          learner: 'logistic'
          C: choose_float(0.01, 10.0, log=True, name='C')
          string__n_components: choose_int(5, 20, name='string__n_components')
        - k: choose_int(15, 25, name='k')
          learner: 'hgb'
          learning_rate: choose_float(0.01, 0.9, log=True, name='learning_rate')
          minhash__n_components: choose_int(10, 30, name='minhash__n_components')

        In the grid above, we see that we have 2 different families of
        configurations: one for the logistic regression and one for the
        gradient boosting. For example we do not have entries with a value for
        both ``C`` and ``minhash__n_components`` because the logistic
        regression and the minhash are never used together.

        The keys in the provided mapping must correspond to the possible
        outcomes of the choice. However, some can be omitted when a default is
        provided.

        >>> learner_kind.match({'logistic': 'linear', 'hgb': 'tree'}).outcome_mapping
        {'logistic': 'linear', 'hgb': 'tree'}
        >>> learner_kind.match({'logistic': 'linear'}, default='unknown').outcome_mapping
        {'logistic': 'linear', 'hgb': 'unknown'}
        """  # noqa : E501
        _check_match_keys(self.outcomes, outcome_mapping.keys(), default is not NULL)
        if default is NULL:
            complete_mapping = outcome_mapping
        else:
            complete_mapping = {
                outcome: outcome_mapping.get(outcome, default)
                for outcome in self.outcomes
            }
        return Match(self, complete_mapping)

    def __repr__(self):
        if self.outcome_names is None:
            arg = self.outcomes
        else:
            arg = {name: out for name, out in zip(self.outcome_names, self.outcomes)}
        args_r = _utils.repr_args((arg,), {"name": self.name})
        return f"choose_from({args_r})"


@dataclasses.dataclass
class Match:
    """Represent the output of a choice ``.match()``"""

    choice: Choice
    outcome_mapping: dict

    __hash__ = None

    def match(self, outcome_mapping, default=NULL):
        """Select a value depending on the result of this match.

        This allows chaining matches. See the docstring of ``Choice.match`` for
        details about match.

        Parameters
        ----------
        outcome_mapping : dict
            Maps possible outcome to the desired result. The keys must all be
            one of the possible outcomes for this ``Match``. If no ``default`` is
            provided, there must be an entry in ``outcome_mapping`` for each
            possible outcome.

        default : object, optional
            The value to use for outcomes not found in ``outcome_mapping``.

        Returns
        -------
        Match
            An object which evaluates to the image of the match through
            ``outcome_mapping``.

        Examples
        --------
        >>> import skrub

        >>> learner_kind = skrub.choose_from(["logistic", "random_forest", "hgb"], name="learner")
        >>> is_linear = learner_kind.match({"logistic": True}, default=False)
        >>> is_linear.outcome_mapping
        {'logistic': True, 'random_forest': False, 'hgb': False}

        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = is_linear.match({True: StandardScaler(), False: 'passthrough'})
        >>> scaler.outcome_mapping
        {'logistic': StandardScaler(), 'random_forest': 'passthrough', 'hgb': 'passthrough'}
        """  # noqa: E501
        _check_match_keys(
            self.outcome_mapping.values(),
            outcome_mapping.keys(),
            default is not NULL,
        )
        mapping = {
            k: outcome_mapping.get(v, default) for k, v in self.outcome_mapping.items()
        }
        return self.choice.match(mapping)

    def as_expr(self):
        """Wrap the match in an expression.

        ``match.as_expr()`` is a convenience shorthand for ``skrub.as_expr(match)``.
        """
        from ._expressions import as_expr

        return as_expr(self)


def choose_from(outcomes, *, name):
    """Construct a choice among several possible outcomes.

    Outcomes can be provided in a list:

    >>> from skrub import choose_from
    >>> choose_from([1, 2], name='the number')
    choose_from([1, 2], name='the number')

    They can also be provided in a dictionary to give a name to each outcome:

    >>> choose_from({'one': 1, 'two': 2}, name='the number')
    choose_from({'one': 1, 'two': 2}, name='the number')
    """
    if isinstance(outcomes, typing.Mapping):
        outcome_names, outcomes = list(outcomes.keys()), list(outcomes.values())
    else:
        outcome_names = None
    return Choice(outcomes, outcome_names=outcome_names, name=name)


def get_default(obj):
    """Extract a value from a Choice, Match, or plain value.

    If the input is a Choice, the default outcome is used.
    Otherwise returns the input.

    >>> from skrub._expressions._choosing import choose_from, get_default
    >>> choice = choose_from([1, 2], name='N')
    >>> choice
    choose_from([1, 2], name='N')
    >>> choice.default()
    1
    >>> get_default(choice)
    1
    >>> get_default(choice.default())
    1
    >>> get_default(1)
    1
    """
    if isinstance(obj, Match):
        return obj.outcome_mapping[get_default(obj.choice)]
    if isinstance(obj, BaseChoice):
        return obj.default()
    return obj


def get_chosen_or_default(obj):
    if isinstance(obj, Match):
        return obj.outcome_mapping[get_chosen_or_default(obj.choice)]
    if isinstance(obj, BaseChoice):
        return obj.chosen_outcome_or_default()
    return obj


class Optional(Choice):
    """A choice between something and nothing."""

    def __repr__(self):
        args = _utils.repr_args(
            (get_default(self),), {"name": self.name}, defaults={"name": None}
        )
        return f"optional({args})"


def optional(value, *, name):
    """Construct a choice between a value and ``None``.

    This is useful for optional steps in a pipeline. If we want to try our
    pipeline with or without dimensionality reduction, we can add a step such
    as:

    >>> from sklearn.decomposition import PCA
    >>> from skrub import optional
    >>> optional(PCA(), name='use dim reduction')
    optional(PCA(), name='use dim reduction')

    The constructed parameter grid will include a version of the pipeline with
    the PCA and one without.
    """
    # TODO remove outcome names
    return Optional([value, None], outcome_names=["true", "false"], name=name)


class BoolChoice(Choice):
    def __repr__(self):
        args = _utils.repr_args((), {"name": self.name})
        return f"choose_bool({args})"

    def if_else(self, if_true, if_false):
        """Select a value depending on this choice.

        This is a convenience shorthand for using ``match``:
        ``boolchoice.if_else(a, b)`` is like
        ``boolchoice.match({True: a, False: b})``.

        See the documentation of ``Choice.match()`` for details.

        Parameters
        ----------
        if_true : object
            Result when this choice selects True.

        if_false : object
            Result when this choice selects False

        Returns
        -------
        Match
            The result corresponding to the value of this choice.
        """
        return self.match({True: if_true, False: if_false})


def choose_bool(*, name):
    """Construct a choice between False and True."""
    return BoolChoice([True, False], outcome_names=None, name=name)


def _check_bounds(low, high, log):
    if high < low:
        raise ValueError(
            f"'high' must be greater than 'low', got low={low}, high={high}"
        )
    if log and low <= 0:
        raise ValueError(f"To use log space 'low' must be > 0, got low={low}")


def _repr_numeric_choice(choice):
    args = _utils.repr_args(
        (choice.low, choice.high),
        {
            "log": choice.log,
            "n_steps": getattr(choice, "n_steps", None),
            "name": choice.name,
        },
        defaults={"log": False, "n_steps": None},
    )
    if choice.to_int:
        return f"choose_int({args})"
    return f"choose_float({args})"


class BaseNumericChoice(BaseChoice):
    """Base class for numeric choices."""

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
    name: str
    chosen_outcome: typing.Optional[typing.Union[int, float]] = None

    def __post_init__(self):
        _check_name(self.name)
        _check_bounds(self.low, self.high, self.log)
        if self.log:
            self._distrib = stats.loguniform(self.low, self.high)
        else:
            self._distrib = stats.uniform(self.low, self.high)

    def rvs(self, size=None, random_state=None):
        value = self._distrib.rvs(size=size, random_state=random_state)
        if self.to_int:
            value = value.astype(int)
        return value

    def default(self):
        low, high = self.low, self.high
        if self.log:
            low, high = np.log(low), np.log(high)
        midpoint = np.mean([low, high])
        if self.log:
            midpoint = np.exp(midpoint)
        if self.to_int:
            midpoint = np.round(midpoint).astype(int)
        return midpoint

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
    name: str
    chosen_outcome: typing.Optional[typing.Union[int, float]] = None

    def __post_init__(self):
        _check_name(self.name)
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
            self.grid = np.unique(np.round(self.grid).astype(int))

    def rvs(self, size=None, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.choice(self.grid, size=size)

    def default(self):
        return self.grid[(len(self.grid) - 1) // 2]

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


def choose_float(low, high, *, log=False, n_steps=None, name):
    """Construct a choice of floating-point numbers from a numeric range."""
    if n_steps is None:
        return NumericChoice(low, high, log=log, to_int=False, name=name)
    return DiscretizedNumericChoice(
        low, high, log=log, to_int=False, n_steps=n_steps, name=name
    )


def choose_int(low, high, *, log=False, n_steps=None, name):
    """Construct a choice of integers from a numeric range."""
    if n_steps is None:
        return NumericChoice(low, high, log=log, to_int=True, name=name)
    return DiscretizedNumericChoice(
        low, high, log=log, to_int=True, n_steps=n_steps, name=name
    )
