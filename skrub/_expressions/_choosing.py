import dataclasses
import functools
import io
import typing
from collections.abc import Sequence

import numpy as np
from scipy import stats
from sklearn.utils import check_random_state

from .. import _utils
from ._utils import Constants


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
    """A base class for all kinds of choices (enumerated, numeric range, ...)

    The main reason they all derive from this base class is to make them easily
    recognizable with ``isinstance`` checks.
    """

    __hash__ = None

    def as_expr(self):
        """Wrap the choice in an expression.

        ``choice.as_expr()`` is a convenience shorthand for ``skrub.as_expr(choice)``.
        """
        from ._expressions import as_expr

        return as_expr(self)

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

        >>> from skrub import choose_from
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
        outcomes = [_with_fields(out, value=func(out.value)) for out in self.outcomes]
        return _with_fields(self, outcomes=outcomes)

    def default(self):
        """Default outcome: the first one in the list."""
        return self.outcomes[0]

    def chosen_outcome_or_default(self):
        if self.chosen_outcome_idx is not None:
            return self.outcomes[self.chosen_outcome_idx]
        return self.default()

    def match(self, outcome_mapping, default=Constants.NO_VALUE):
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
        values = [unwrap(outcome) for outcome in self.outcomes]
        _check_match_keys(
            values, outcome_mapping.keys(), default is not Constants.NO_VALUE
        )
        if default is Constants.NO_VALUE:
            complete_mapping = outcome_mapping
        else:
            complete_mapping = {
                outcome: outcome_mapping.get(outcome, default) for outcome in values
            }
        return Match(self, complete_mapping)

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
    """Represent the output of a choice ``.match()``"""

    choice: Choice
    outcome_mapping: dict

    __hash__ = None

    def match(self, outcome_mapping, default=Constants.NO_VALUE):
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
            default is not Constants.NO_VALUE,
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
        prepared_outcomes = [Outcome(val, key) for key, val in outcomes.items()]
    else:
        prepared_outcomes = [Outcome(val) for val in outcomes]
    return Choice(prepared_outcomes, name=name)


def unwrap(obj):
    """Extract the value from an Outcome or a plain value.

    If the input is a plain value, it is returned unchanged.

    >>> from skrub._expressions._choosing import choose_from, unwrap
    >>> choice = choose_from([1, 2], name='N')
    >>> outcome = choice.default()
    >>> outcome
    Outcome(value=1, name=None, in_choice='N')
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

    >>> from skrub._expressions._choosing import choose_from, unwrap_default
    >>> choice = choose_from([1, 2], name='N')
    >>> choice
    choose_from([1, 2], name='N')
    >>> choice.default()
    Outcome(value=1, name=None, in_choice='N')
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
    return Optional([Outcome(value, "true"), Outcome(None, "false")], name=name)


class BoolChoice(Choice):
    def __repr__(self):
        args = _utils.repr_args((), {"name": self.name}, defaults={"name": None})
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
        return self._wrap_outcome(self.grid[item])

    def __len__(self):
        return len(self.grid)

    def __iter__(self):
        return iter(map(self._wrap_outcome, self.grid))


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


def expand_grid(grid):
    from ._evaluation import param_grid

    return param_grid(grid)


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
