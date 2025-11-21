import dataclasses
import functools
import numbers
import typing
from collections.abc import Sequence

import numpy as np
from scipy import stats
from sklearn.utils import check_random_state

from .. import _utils
from ._utils import NULL, OPTIONAL_VALUE


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

    def as_data_op(self):
        """Wrap the choice in a DataOp.

        `choice.as_data_op()`` is a convenience shorthand for
        ``skrub.as_data_op(choice)``.
        """
        from ._data_ops import as_data_op

        return as_data_op(self)

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


def _check_match_keys(outcomes, match_keys, match_has_default):
    try:
        extra_keys = set(match_keys).difference(outcomes)
        extra_outcomes = set(outcomes).difference(match_keys)
    except TypeError as e:
        raise TypeError(f"To use `match()`, all choice outcomes must be hashable. {e}")
    if extra_keys:
        raise ValueError(
            "The following keys were found in the mapping provided to `match()` but"
            f" are not possible choice outcomes: {extra_keys!r}"
        )
    if match_has_default:
        return
    if extra_outcomes:
        raise ValueError(
            "The following outcomes do not have a corresponding key in the mapping"
            f" provided to `match()`: {extra_outcomes!r}. Please provide an entry for"
            " each possible outcome, or a default."
        )


def _check_name(name):
    if name is not None and not isinstance(name, str):
        raise TypeError(
            f"choice `name` must be a `str` or `None`, got object of type: {type(name)}"
        )


class _Ellipsis:
    def __repr__(self):
        return "…"


@dataclasses.dataclass
class Choice(BaseChoice):
    """A choice among an enumerated set of outcomes."""

    outcomes: list[typing.Any]
    outcome_names: typing.Optional[list[str]]
    name: typing.Optional[str] = None
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

    def default(self):
        """Get the default outcome."""
        return self.outcomes[0]

    def chosen_outcome_or_default(self):
        """Chosen outcome when it has been set, otherwise the default."""
        if self.chosen_outcome_idx is not None:
            return self.outcomes[self.chosen_outcome_idx]
        return self.default()

    def match(self, outcome_mapping, default=NULL):
        """Select a value depending on the outcome of the choice.

        This allows inserting several inter-dependent hyper-parameters in a
        DataOps plan (and the resulting learner), which all depend on the outcome
        of the same choice. See the examples below.

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
        supervised estimator we use. the :class:`MinHashEncoder` can be a good
        choice when the downstream estimator is a tree-based model, but not when
        it is a linear model. So we have 2 choices in our DataOps plan, the
        encoder and the estimator, but they are not independent: not all
        combinations make sense.

        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import HistGradientBoostingClassifier
        >>> from sklearn.feature_selection import SelectKBest
        >>> import skrub

        >>> estimator_kind = skrub.choose_from(["logistic", "hgb"], name="estimator")
        >>> encoder = estimator_kind.match(
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
        >>> predictor = estimator_kind.match(
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
          estimator: 'logistic'
          string__n_components: choose_int(5, 20, name='string__n_components')
          C: choose_float(0.01, 10.0, log=True, name='C')
        - k: choose_int(15, 25, name='k')
          estimator: 'hgb'
          minhash__n_components: choose_int(10, 30, name='minhash__n_components')
          learning_rate: choose_float(0.01, 0.9, log=True, name='learning_rate')

        In the grid above, we see that we have 2 different families of
        configurations: one for the logistic regression and one for the
        gradient boosting. For example we do not have entries with a value for
        both ``C`` and ``minhash__n_components`` because the logistic
        regression and the minhash are never used together.

        The keys in the provided mapping must correspond to the possible
        outcomes of the choice. However, some can be omitted when a default is
        provided.

        >>> estimator_kind.match({'logistic': 'linear', 'hgb': 'tree'}).outcome_mapping
        {'logistic': 'linear', 'hgb': 'tree'}
        >>> estimator_kind.match({'logistic': 'linear'}, default='unknown').outcome_mapping
        {'logistic': 'linear', 'hgb': 'unknown'}
        """  # noqa : E501
        _check_match_keys(
            outcomes=self.outcomes,
            match_keys=outcome_mapping.keys(),
            match_has_default=default is not NULL,
        )
        complete_mapping = {
            outcome: outcome_mapping.get(outcome, default) for outcome in self.outcomes
        }
        return Match(self, complete_mapping)

    def __repr__(self):
        if self.outcome_names is None:
            arg = self.outcomes
        else:
            arg = dict(zip(self.outcome_names, self.outcomes))
        args_r = _utils.repr_args((arg,), {"name": self.name}, {"name": None})
        return f"choose_from({args_r})"

    def __skrub_short_repr__(self):
        if self.outcome_names is None:
            return repr(self)
        arg = {
            name: _Ellipsis() for name, out in zip(self.outcome_names, self.outcomes)
        }
        args_r = _utils.repr_args((arg,), {"name": self.name}, {"name": None})
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

        >>> estimator_kind = skrub.choose_from(["logistic", "random_forest", "hgb"], name="estimator")
        >>> is_linear = estimator_kind.match({"logistic": True}, default=False)
        >>> is_linear.outcome_mapping
        {'logistic': True, 'random_forest': False, 'hgb': False}

        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = is_linear.match({True: StandardScaler(), False: 'passthrough'})
        >>> scaler.outcome_mapping
        {'logistic': StandardScaler(), 'random_forest': 'passthrough', 'hgb': 'passthrough'}
        """  # noqa: E501
        _check_match_keys(
            outcomes=self.outcome_mapping.values(),
            match_keys=outcome_mapping.keys(),
            match_has_default=default is not NULL,
        )
        mapping = {
            k: outcome_mapping.get(v, default) for k, v in self.outcome_mapping.items()
        }
        return self.choice.match(mapping)

    def as_data_op(self):
        """Wrap the match in a DataOp.

        ``match.as_data_op()`` is a convenience shorthand for
        ``skrub.as_data_op(match)``.
        """
        from ._data_ops import as_data_op

        return as_data_op(self)


def choose_from(outcomes, *, name=None):
    """A choice among several possible outcomes.

    When a learner is used *without hyperparameter tuning*, the outcome of
    this choice is the first value in the ``outcomes`` list or dict.

    Parameters
    ----------
    outcomes : list or dict
        The possible outcomes to choose from. If a dict, the values are the
        outcomes and the keys give them human-readable names used to display
        hyperparameter search grids and results.

    name : str, optional (default=None)
        If not ``None``, ``name`` is used when displaying search results and
        can also be used to override the choice's value by setting it in the
        environment containing a learner's inputs.

    Returns
    -------
    Choice
        An object representing this choice, which can be used in a skrub
        DataOps plan.

    See also
    --------
    choose_bool :
        Construct a choice between False and True.

    choose_float :
        Construct a choice of floating-point numbers from a numeric range.

    choose_int :
        Construct a choice of integers from a numeric range.

    Examples
    --------
    Outcomes can be provided in a list:

    >>> from skrub import choose_from
    >>> choose_from([1, 2], name='the number')
    choose_from([1, 2], name='the number')

    They can also be provided in a dictionary to give a name to each outcome:

    >>> choose_from({'one': 1, 'two': 2}, name='the number')
    choose_from({'one': 1, 'two': 2}, name='the number')

    When a skrub learner containing a ``choose_from`` is fitted *without
    hyperparameter tuning*, the default outcome for the choice is used.
    It is the first outcome in the provided list or dict:

    >>> choose_from([1, 2, 3]).default()
    1
    """
    if isinstance(outcomes, typing.Mapping):
        outcome_names, outcomes = list(outcomes.keys()), list(outcomes.values())
    else:
        outcome_names = None
    return Choice(outcomes, outcome_names=outcome_names, name=name)


# helper for other skrub modules
def get_default(obj):
    """Extract the default value from a Choice, Match, or plain value.

    If the input is a Choice, the default outcome is used.
    Otherwise returns the input.

    >>> from skrub._data_ops._choosing import choose_from, get_default
    >>> choice = choose_from([1, 2], name='N')
    >>> choice
    choose_from([1, 2], name='N')
    >>> choice.default()
    1
    >>> get_default(choice)
    1
    >>> get_default(choice.match({1: 'one', 2: 'two'}))
    'one'
    >>> get_default(1)
    1
    """
    if isinstance(obj, Match):
        return obj.outcome_mapping[get_default(obj.choice)]
    if isinstance(obj, BaseChoice):
        return obj.default()
    return obj


# helper for other skrub modules
def get_chosen_or_default(obj):
    """Extract the chosen (or if not set, default) value."""
    if isinstance(obj, Match):
        return obj.outcome_mapping[get_chosen_or_default(obj.choice)]
    if isinstance(obj, BaseChoice):
        return obj.chosen_outcome_or_default()
    return obj


# helper for other skrub modules
def get_display_name(choice):
    """Name to use for representing a choice in CV results and plots."""
    if choice.name is not None:
        return choice.name
    return _utils.short_repr(choice)


class Optional(Choice):
    """A choice between something and nothing."""

    def __repr__(self):
        if self.outcomes[0] is not None or self.outcomes[1] is None:
            # note when `value` is None, `default` makes no difference
            value, default = self.outcomes[0], OPTIONAL_VALUE
        else:
            value, default = self.outcomes[1], None
        args = _utils.repr_args(
            (value,),
            {"name": self.name, "default": default},
            defaults={"name": None, "default": OPTIONAL_VALUE},
        )
        return f"optional({args})"


def optional(value, *, name=None, default=OPTIONAL_VALUE):
    """A choice between ``value`` and ``None``.

    When a learner is fitted *without hyperparameter tuning*, the outcome of
    this choice is ``value``. Pass ``default=None`` to make ``None`` the
    default outcome.

    Parameters
    ----------
    value : object
        The outcome (when ``None`` is not chosen).

    name : str, optional (default=None)
        If not ``None``, ``name`` is used when displaying search results and
        can also be used to override the choice's value by setting it in the
        environment containing a learner's inputs.

    default : NoneType, optional
        An ``optional`` is a choice between the provided ``value`` and
        ``None``. Normally, the default outcome when a learner is used
        *without hyperparameter tuning* is the provided ``value``. Pass
        ``default=None`` to make the alternative outcome, ``None``, the
        default. ``None`` is the only allowed value for this parameter.

    Returns
    -------
    Choice
        An object representing this choice, which can be used in a skrub
        learner.

    Examples
    --------
    ``optional`` is useful for optional steps in a DataOps plan, or a learner.
    If we want to try a learner with or without dimensionality reduction, we can
    add a step such as:

    >>> from sklearn.decomposition import PCA
    >>> from skrub import optional
    >>> optional(PCA(), name='use dim reduction')
    optional(PCA(), name='use dim reduction')

    The constructed parameter grid will include a branch of the plan with the
    the PCA and one without:

    >>> print(
    ... optional(PCA(), name='dim reduction').as_data_op().skb.describe_param_grid()
    ... )
    - dim reduction: [PCA(), None]

    When a learner that contains an ``optional`` step is used *without
    hyperparameter tuning*, the default outcome is the provided ``value``.

    >>> print(optional(PCA()).default())
    PCA()

    This can be overridden by passing ``default=None``:

    >>> print(optional(PCA(), default=None).default())
    None
    """
    if default is not OPTIONAL_VALUE and default is not None:
        raise TypeError(
            "If provided, the `default` argument must be `None`. "
            f"Got object of type: {type(default)}"
        )
    outcomes = [None, value] if default is None else [value, None]
    return Optional(outcomes, outcome_names=None, name=name)


class BoolChoice(Choice):
    def __repr__(self):
        args = _utils.repr_args(
            (),
            {"name": self.name, "default": self.outcomes[0]},
            {"name": None, "default": True},
        )
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


def choose_bool(*, name=None, default=True):
    """A choice between ``True`` and ``False``.

    When a learner is fitted *without hyperparameter tuning*, the outcome of
    this choice is ``True``. Pass ``default=False`` to make ``False`` the
    default outcome.

    Parameters
    ----------
    name : str, optional (default=None)
        If not ``None``, ``name`` is used when displaying search results and
        can also be used to override the choice's value by setting it in the
        environment containing a learner's inputs.

    default : bool, optional (default=True)
        Choice's default value when hyperparameter search is not used.

    Returns
    -------
    BoolChoice
        An object representing this choice, which can be used in a skrub
        learner.

    See also
    --------
    choose_float :
        Construct a choice of floating-point numbers from a numeric range.

    choose_from :
        Construct a choice among several possible outcomes.

    choose_int :
        Construct a choice of integers from a numeric range.

    Examples
    --------
    >>> import skrub
    >>> print(skrub.choose_bool().as_data_op().skb.describe_param_grid())
    - choose_bool(): [True, False]
    >>> skrub.choose_bool().default()
    True

    We can set the default to make it ``False``:

    >>> skrub.choose_bool(default=False).default()
    False
    """
    default = bool(default)
    return BoolChoice([default, not default], outcome_names=None, name=name)


def _check_bounds(low, high, log):
    if high < low:
        raise ValueError(
            f"'high' must be greater than 'low', got low={low}, high={high}"
        )
    if log and low <= 0:
        raise ValueError(f"To use log space 'low' must be > 0, got low={low}")


def _check_default_numeric_outcome(default, to_int):
    if default is None:
        return
    if to_int and not isinstance(default, numbers.Integral):
        raise TypeError(
            "The default for `choose_int` must be an integer. "
            f"Got object of type: {type(default)}"
        )
    if not isinstance(default, numbers.Real):
        raise TypeError(
            "The default for `choose_float` must be a float. "
            f"Got object of type: {type(default)}"
        )


def _repr_numeric_choice(choice):
    args = _utils.repr_args(
        (choice.low, choice.high),
        {
            "log": choice.log,
            "n_steps": getattr(choice, "n_steps", None),
            "name": choice.name,
            "default": choice.default_outcome,
        },
        defaults={"log": False, "n_steps": None, "name": None, "default": None},
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
    default_outcome: float
    chosen_outcome: typing.Optional[typing.Union[int, float]] = None

    def __post_init__(self):
        _check_name(self.name)
        _check_bounds(self.low, self.high, self.log)
        _check_default_numeric_outcome(self.default_outcome, self.to_int)

        # if choose_int, sampled values will be truncated to produce int so we
        # sample in [ low, high+1 [ (otherwise high would be excluded from the
        # range)
        offset = 1 - 1e-6 if self.to_int else 0

        if self.log:
            # loguniform(a, b).support() -> (a, b)
            self._distrib = stats.loguniform(self.low, self.high + offset)
        else:
            # uniform(a, b).support() -> (a, a + b)
            self._distrib = stats.uniform(self.low, self.high - self.low + offset)

    def rvs(self, size=None, random_state=None):
        value = self._distrib.rvs(size=size, random_state=random_state)
        if self.to_int:
            value = value.astype(int)
        return value

    def default(self):
        if self.default_outcome is not None:
            return self.default_outcome
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
    default_outcome: float
    chosen_outcome: typing.Optional[typing.Union[int, float]] = None

    def __post_init__(self):
        _check_name(self.name)
        _check_bounds(self.low, self.high, self.log)
        _check_default_numeric_outcome(self.default_outcome, self.to_int)
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
        if self.default_outcome is not None:
            return self.default_outcome
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


def choose_float(low, high, *, log=False, n_steps=None, name=None, default=None):
    """A choice of floating-point numbers from a numeric range.

    When a learner is fitted *without hyperparameter tuning*, the outcome of
    this choice is the middle of the range (possibly on a ``log`` scale). Pass
    a float as the ``default`` argument to set the default outcome.

    Parameters
    ----------
    low : float
        The start of the range.

    high : float
        Then end of the range.

    log : bool, optional (default=False)
        Whether sampling should be done on a logarithmic scale.

    n_steps : int, optional (default=None)
        If not ``None``, a grid of ``n_steps`` values across the range is
        defined and sampling is done on that grid. This can be useful to limit
        the number of unique values that get sampled, for example to improve
        the effectiveness of caching. However, it means a much more restricted
        space of possible values gets explored.

    name : str, optional (default=None)
        If not ``None``, ``name`` is used when displaying search results and
        can also be used to override the choice's value by setting it in the
        environment containing a learner's inputs.

    default : float, optional (default=None)
        If provided, override the choice's default value when hyperparameter
        search is not used. Otherwise the default value is the middle of the
        range (either on a linear or logarithmic scale depending on the value
        of ``log``).

    Returns
    -------
    numeric choice
        An object representing this choice, which can be used in a skrub
        learner.

    See also
    --------
    choose_bool :
        Construct a choice between False and True.

    choose_int :
        Construct a choice of integers from a numeric range.

    choose_from :
        Construct a choice among several possible outcomes.
    """
    if n_steps is None:
        return NumericChoice(
            low, high, log=log, to_int=False, name=name, default_outcome=default
        )
    return DiscretizedNumericChoice(
        low,
        high,
        log=log,
        to_int=False,
        n_steps=n_steps,
        name=name,
        default_outcome=default,
    )


def choose_int(low, high, *, log=False, n_steps=None, name=None, default=None):
    """A choice of integers from a numeric range.

    When a learner is fitted *without hyperparameter tuning*, the outcome of
    this choice is the middle of the range (possibly on a ``log`` scale). Pass
    an int as the ``default`` argument to set the default outcome.

    Parameters
    ----------
    low : int
        The start of the range.

    high : int
        Then end of the range. It is included in the possible outcomes.

    log : bool, optional (default=False)
        Whether sampling should be done on a logarithmic scale.

    n_steps : int, optional (default=None)
        If not ``None``, a grid of ``n_steps`` values across the range is
        defined and sampling is done on that grid. This can be useful to limit
        the number of unique values that get sampled, for example to improve
        the effectiveness of caching. However, it means a much more restricted
        space of possible values gets explored.

    name : str, optional (default=None)
        If not ``None``, ``name`` is used when displaying search results and
        can also be used to override the choice's value by setting it in the
        environment containing a learner's inputs.

    default : int, optional (default=None)
        If provided, override the choice's default value when hyperparameter
        search is not used. Otherwise the default value is the integer closest
        to the middle of the range (either on a linear or logarithmic scale
        depending on the value of ``log``).

    Returns
    -------
    numeric choice
        An object representing this choice, which can be used in a skrub
        learner.

    See also
    --------
    choose_bool :
        Construct a choice between False and True.

    choose_float :
        Construct a choice of floating-point numbers from a numeric range.

    choose_from :
        Construct a choice among several possible outcomes.
    """
    if n_steps is None:
        return NumericChoice(
            low, high, log=log, to_int=True, name=name, default_outcome=default
        )
    return DiscretizedNumericChoice(
        low,
        high,
        log=log,
        to_int=True,
        n_steps=n_steps,
        name=name,
        default_outcome=default,
    )
