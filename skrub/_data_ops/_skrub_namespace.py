import pickle
import typing
import warnings

from sklearn import model_selection

from .. import selectors as s
from .._select_cols import DropCols, SelectCols
from ._data_ops import (
    AppliedEstimator,
    Apply,
    Concat,
    DataOp,
    FreezeAfterFit,
    IfElse,
    Match,
    Var,
    check_data_op,
    check_name,
    deferred,
)
from ._estimator import (
    ParamSearch,
    SkrubLearner,
    cross_validate,
    iter_cv_splits,
    train_test_split,
)
from ._evaluation import (
    choices,
    clone,
    describe_steps,
    evaluate,
    nodes,
)
from ._inspection import (
    describe_param_grid,
    draw_data_op_graph,
    full_report,
)
from ._subsampling import SubsamplePreviews, env_with_subsampling
from ._utils import KFOLD_5, NULL, attribute_error

_SELECT_ALL_COLUMNS = s.all()


def _var_values_provided(data_op, environment):
    all_nodes = nodes(data_op)
    names = {
        node._skrub_impl.name for node in all_nodes if isinstance(node._skrub_impl, Var)
    }
    intersection = names.intersection(environment.keys())
    return bool(intersection)


def _check_keep_subsampling(fitted, keep_subsampling):
    if not fitted and keep_subsampling:
        raise ValueError(
            "Subsampling is only applied when fitting the estimator "
            "on the data already provided when initializing variables. "
            "Please pass `fitted=True` or `keep_subsampling=False`."
        )


def _check_can_be_pickled(obj):
    try:
        dumped = pickle.dumps(obj)
        pickle.loads(dumped)
    except Exception as e:
        msg = "The check to verify that the learner can be serialized failed."
        if "recursion" in str(e).lower():
            msg = (
                f"{msg} Is a step in the learner holding a reference to "
                "the full learner itself? For example a global variable "
                "in a `@skrub.deferred` function?"
            )
        raise pickle.PicklingError(msg) from e


def _check_grid_search_possible(data_op):
    for c in choices(data_op).values():
        if hasattr(c, "rvs") and not isinstance(c, typing.Sequence):
            raise ValueError(
                "Cannot use grid search with continuous numeric ranges. "
                "Please use `make_randomized_search` or provide a number "
                f"of steps for this range: {c}"
            )


class SkrubNamespace:
    """The data_ops' ``.skb`` attribute."""

    # NOTE: if some DataOps types are given additional methods not
    # available for all DataOps (eg if some methods specific to
    # ``.skb.apply()`` nodes are added), we can create new namespace classes
    # derived from this one and return the appropriate one in
    # ``_data_ops._Skb.__get__``.

    def __init__(self, data_op):
        self._data_op = data_op

    def _apply(
        self,
        estimator,
        y=None,
        cols=_SELECT_ALL_COLUMNS,
        how="auto",
        allow_reject=False,
        unsupervised=False,
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}
        data_op = DataOp(
            Apply(
                estimator=estimator,
                cols=cols,
                X=self._data_op,
                y=y,
                how=how,
                allow_reject=allow_reject,
                unsupervised=unsupervised,
                kwargs=kwargs,
            )
        )
        return data_op

    @check_data_op
    def apply(
        self,
        estimator,
        *,
        y=None,
        cols=_SELECT_ALL_COLUMNS,
        exclude_cols=None,
        how="auto",
        allow_reject=False,
        unsupervised=False,
        fit_kwargs=None,
        fit_transform_kwargs=None,
        transform_kwargs=None,
        predict_kwargs=None,
        predict_proba_kwargs=None,
        decision_function_kwargs=None,
        score_kwargs=None,
    ):
        """
        Apply a scikit-learn estimator to a dataframe or numpy array.

        Parameters
        ----------
        estimator : scikit-learn estimator
            The transformer or predictor to apply.

        y : dataframe, column or numpy array, optional
            The prediction targets when ``estimator`` is a supervised estimator.

        cols : string, list of strings or skrub selector, optional
            The columns to transform, when ``estimator`` is a transformer.

        exclude_cols : string, list of strings or skrub selector, optional
            When ``estimator`` is a transformer, columns to which it should
            _not_ be applied. The columns that are matched by ``cols`` AND not
            matched by ``exclude_cols`` are transformed.

        how : "auto", "cols", "frame" or "no_wrap", optional
            How the estimator is applied. In most cases the default "auto"
            is appropriate.
            - "cols" means `estimator` is wrapped in a :class:`ApplyToCols`
              transformer, which fits a separate clone of `estimator` each
              column in `cols`. `estimator` must be a transformer (have a
              ``fit_transform`` method).
            - "frame" means `estimator` is wrapped in a :class:`ApplyToFrame`
              transformer, which fits a single clone of `estimator` to the
              selected part of the input dataframe. `estimator` must be a
              transformer.
            - "no_wrap" means no wrapping, `estimator` is applied directly to
              the unmodified input.
            - "auto" chooses the wrapping depending on the input and estimator.
              If the input is not a dataframe or the estimator is not a
              transformer, the "no_wrap" strategy is chosen. Otherwise if the
              estimator has a ``__single_column_transformer__`` attribute,
              "cols" is chosen. Otherwise "frame" is chosen.

        allow_reject : bool, optional
            Whether the transformer can refuse to transform columns for which
            it does not apply, in which case they are passed through unchanged.
            This can be useful to avoid specifying exactly which columns should
            be transformed. For example if we apply ``skrub.ToDatetime()`` to
            all columns with ``allow_reject=True``, string columns that can be
            parsed as dates will be converted and all other columns will be
            passed through. If we use ``allow_reject=False`` (the default), an
            error would be raised if the dataframe contains columns for which
            ``ToDatetime`` does not apply (eg a column of numbers).

        unsupervised : bool, optional
            Use this to indicate that ``y`` is required for scoring but not
            fitting, as is the case for clustering algorithms. If ``y`` is not
            required at all (for example when applying an unsupervised
            transformer, or when we are not interested in scoring with
            ground-truth labels), simply leave the default ``y=None`` and there
            is no need to pass a value for ``unsupervised``.

        fit_kwargs : dict, optional, default=None
            Extra named arguments to pass to the estimator's ``fit()`` method,
            for example ``fit_kwargs={'sample_weights': [.1, .5, .4]}``. May be
            (or contain) a DataOp, which will be evaluated before passing the
            kwargs to ``fit``.
        fit_transform_kwargs : dict, optional, default=None
            Extra named arguments for ``fit_transform``. See the description of
            the ``fit_kwargs`` parameter.
        transform_kwargs : dict, optional, default=None
            Extra named arguments for ``transform``. See the description of the
            ``fit_kwargs`` parameter.
        predict_kwargs : dict, optional, default=None
            Extra named arguments for ``predict``. See the description of the
            ``fit_kwargs`` parameter.
        predict_proba_kwargs : dict, optional, default=None
            Extra named arguments for ``predict_proba``. See the description of
            the ``fit_kwargs`` parameter.
        decision_function_kwargs : dict, optional, default=None
            Extra named arguments for ``decision_function``. See the
            description of the ``fit_kwargs`` parameter.
        score_kwargs : dict, optional, default=None
            Extra named arguments for ``score``. See the description of the
            ``fit_kwargs`` parameter.

        Returns
        -------
        result
            The transformed dataframe when ``estimator`` is a transformer, and
            the fitted ``estimator``'s predictions if it is a supervised
            predictor.

        See also
        --------
        skrub.DataOp.skb.make_learner :
            Get a skrub learner for this DataOp.
        skrub.ApplyToCols :
            Transformer that applies a given transformer separately to each
            selected column.
        skrub.ApplyToFrame:
            Transformer that applies a given transformer to part of a
            dataframe.

        Examples
        --------
        >>> import skrub
        >>> data = skrub.datasets.toy_orders()
        >>> x = skrub.X(data.X)
        >>> x
        <Var 'X'>
        Result:
        ―――――――
           ID product  quantity        date
        0   1     pen         2  2020-04-03
        1   2     cup         3  2020-04-04
        2   3     cup         5  2020-04-04
        3   4   spoon         1  2020-04-05

        >>> datetime_encoder = skrub.DatetimeEncoder(add_total_seconds=False)
        >>> x.skb.apply(skrub.TableVectorizer(datetime=datetime_encoder))
        <Apply TableVectorizer>
        Result:
        ―――――――
            ID  product_cup  product_pen  ...  date_year  date_month  date_day
        0  1.0          0.0          1.0  ...     2020.0         4.0       3.0
        1  2.0          1.0          0.0  ...     2020.0         4.0       4.0
        2  3.0          1.0          0.0  ...     2020.0         4.0       4.0
        3  4.0          0.0          0.0  ...     2020.0         4.0       5.0

        Transform only the ``'product'`` column:

        >>> x.skb.apply(skrub.StringEncoder(n_components=2), cols='product') # doctest: +SKIP
        <Apply StringEncoder>
        Result:
        ―――――――
           ID     product_0     product_1  quantity        date
        0   1 -2.560113e-16  1.000000e+00         2  2020-04-03
        1   2  1.000000e+00  7.447602e-17         3  2020-04-04
        2   3  1.000000e+00  7.447602e-17         5  2020-04-04
        3   4 -3.955170e-16 -8.326673e-17         1  2020-04-05

        Transform all but the ``'ID'`` and ``'quantity'`` columns:

        >>> x.skb.apply(
        ...     skrub.StringEncoder(n_components=2), exclude_cols=["ID", "quantity"]
        ... ) # doctest: +SKIP
        <Apply StringEncoder>
        Result:
        ―――――――
           ID     product_0     product_1  quantity    date_0    date_1
        0   1  9.775252e-08  7.830415e-01         2  0.766318 -0.406667
        1   2  9.999999e-01  0.000000e+00         3  0.943929  0.330148
        2   3  9.999998e-01 -1.490116e-08         5  0.943929  0.330149
        3   4  9.910963e-08 -6.219692e-01         1  0.766318 -0.406668

        More complex selection of the columns to transform, here all numeric
        columns except the ``'ID'``:

        >>> from sklearn.preprocessing import StandardScaler
        >>> from skrub import selectors as s

        >>> x.skb.apply(StandardScaler(), cols=s.numeric() - "ID")
        <Apply StandardScaler>
        Result:
        ―――――――
           ID product        date  quantity
        0   1     pen  2020-04-03 -0.507093
        1   2     cup  2020-04-04  0.169031
        2   3     cup  2020-04-04  1.521278
        3   4   spoon  2020-04-05 -1.183216

        For supervised estimators, pass the targets as the argument for ``y``:

        >>> from sklearn.dummy import DummyClassifier
        >>> y = skrub.y(data.y)
        >>> y
        <Var 'y'>
        Result:
        ―――――――
        0    False
        1    False
        2     True
        3    False
        Name: delayed, dtype: bool

        >>> x.skb.apply(skrub.TableVectorizer()).skb.apply(DummyClassifier(), y=y)
        <Apply DummyClassifier>
        Result:
        ―――――――
        0    False
        1    False
        2    False
        3    False
        Name: delayed, dtype: bool

        We can also pass additional keyword arguments to the estimator's
        methods. For example a StandardScaler can be passed sample weights.
        We first apply it without weights for comparison:

        >>> import pandas as pd
        >>> X = skrub.var("X", pd.DataFrame({"count": [10, 1], "value": [2.0, -2.0]}))
        >>> count, value = X["count"], X[["value"]]
        >>> value.skb.apply(StandardScaler())
        <Apply StandardScaler>
        Result:
        ―――――――
           value
        0    1.0
        1   -1.0

        Now we weight by ``count``. Note that ``count`` is itself a DataOp -- the
        kwargs, like X and y, can be computed during the DataOp's evaluation:

        >>> value.skb.apply(StandardScaler(), fit_transform_kwargs={"sample_weight": count})
        <Apply StandardScaler>
        Result:
        ―――――――
              value
        0  0.316...
        1 -3.162...

        Another example would be passing evaluation sets to the ``fit`` method
        of an ``xgboost`` estimator.

        Sometimes we want to pass a value for ``y`` because it is required for
        scoring and cross-validation, but it is not needed for fitting the
        estimator. In this case pass ``unsupervised=True``.

        >>> from sklearn.datasets import make_blobs
        >>> from sklearn.cluster import KMeans

        >>> X, y = make_blobs(n_samples=10, random_state=0)
        >>> e = skrub.X(X).skb.apply(
        ...     KMeans(n_clusters=2, n_init=1, random_state=0),
        ...     y=skrub.y(y),
        ...     unsupervised=True,
        ... )
        >>> e.skb.cross_validate()["test_score"]  # doctest: +SKIP
        0   -19.437348
        1   -12.463938
        2   -11.804288
        3   -37.238832
        4    -4.857855
        Name: test_score, dtype: float64
        >>> learner = e.skb.make_learner().fit({"X": X})
        >>> learner.predict({"X": X})  # doctest: +SKIP
        array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=int32)
        """  # noqa: E501
        # TODO later we could also expose `wrap_transformer`'s `keep_original`
        # and `rename_cols` params
        if exclude_cols is not None:
            cols = s.make_selector(cols) - exclude_cols
        # unsupervised should be an actual bool
        unsupervised = bool(unsupervised)
        return self._apply(
            estimator=estimator,
            y=y,
            cols=cols,
            how=how,
            allow_reject=allow_reject,
            unsupervised=unsupervised,
            kwargs={
                "fit": fit_kwargs,
                "fit_transform": fit_transform_kwargs,
                "transform": transform_kwargs,
                "predict": predict_kwargs,
                "predict_proba": predict_proba_kwargs,
                "decision_function": decision_function_kwargs,
                "score": score_kwargs,
            },
        )

    def apply_func(self, func, *args, **kwargs):
        r"""Apply the given function.

        This is a convenience function; ``X.skb.apply_func(func)`` is
        equivalent to ``skrub.deferred(func)(X)``.

        Parameters
        ----------
        func : function
            The function to apply to the DataOp.

        args
            additional positional arguments passed to ``func``.

        kwargs
            named arguments passed to ``func``.

        Returns
        -------
        data_op
            The DataOp that evaluates to the result of calling ``func`` as
            ``func(self, *args, **kwargs)``.

        Examples
        --------
        >>> import skrub

        >>> def count_words(text, sep=None):
        ...     return len(text.split(sep))

        >>> text = skrub.var("text", "Hello, world!")
        >>> text
        <Var 'text'>
        Result:
        ―――――――
        'Hello, world!'

        >>> count = text.skb.apply_func(count_words)
        >>> count
        <Call 'count_words'>
        Result:
        ―――――――
        2
        >>> count.skb.eval({"text": "one two three four"})
        4

        We can pass extra arguments:

        >>> text.skb.apply_func(count_words, sep="\n")
        <Call 'count_words'>
        Result:
        ―――――――
        1

        Using ``.skb.apply_func`` is the same as using ``deferred``, for example:

        >>> skrub.deferred(count_words)(text)
        <Call 'count_words'>
        Result:
        ―――――――
        2
        """
        return deferred(func)(self._data_op, *args, **kwargs)

    @check_data_op
    def if_else(self, value_if_true, value_if_false):
        """Create a conditional DataOp.

        If ``self`` evaluates to ``True``, the result will be
        ``value_if_true``, otherwise ``value_if_false``.

        Parameters
        ----------
        value_if_true
            The value to return if ``self`` is true.

        value_if_false
            The value to return if ``self`` is false.

        Returns
        -------
        Conditional DataOp

        See also
        --------
        skrub.DataOp.skb.match :
            Select based on the value of a DataOp.

        Notes
        -----
        The branch which is not selected is not evaluated, which is the main
        advantage compared to wrapping the conditional statement in a
        ``@skrub.deferred`` function.

        Examples
        --------
        >>> import skrub
        >>> import numpy as np
        >>> a = skrub.var('a')
        >>> shuffle_a = skrub.var('shuffle_a')

        >>> @skrub.deferred
        ... def shuffled(values):
        ...     print('shuffling')
        ...     return np.random.default_rng(0).permutation(values)

        >>> @skrub.deferred
        ... def copy(values):
        ...     print('copying')
        ...     return values.copy()


        >>> b = shuffle_a.skb.if_else(shuffled(a), copy(a))
        >>> b
        <IfElse <Var 'shuffle_a'> ? <Call 'shuffled'> : <Call 'copy'>>

        Note that only one of the 2 branches is evaluated:

        >>> b.skb.eval({'a': np.arange(3), 'shuffle_a': True})
        shuffling
        array([2, 0, 1])
        >>> b.skb.eval({'a': np.arange(3), 'shuffle_a': False})
        copying
        array([0, 1, 2])
        """
        return DataOp(IfElse(self._data_op, value_if_true, value_if_false))

    @check_data_op
    def match(self, targets, default=NULL):
        """Select based on the value of a DataOp.

        Evaluate ``self``, then compare the result to the keys in ``targets``.
        If there is a match, evaluate the corresponding value and return it. If
        there is no match and ``default`` has been provided, evaluate and return
        ``default``. Otherwise, a ``KeyError`` is raised.

        Therefore, only one of the branches or the default is evaluated which
        is the main advantage compared to placing the selection inside a
        ``@skrub.deferred`` function.

        Parameters
        ----------
        targets : dict
            A dictionary providing which result to select. The keys must be
            actual values, they **cannot** be DataOps. The values can be
            DataOps or any object.

        default : object, optional
            If provided, the match falls back to the default when none of the
            targets have matched.

        Returns
        -------
        The value corresponding to the matching key or the default

        See also
        --------
        skrub.deferred :
            Wrap function calls in a :class:`DataOp`.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var("a")
        >>> mode = skrub.var("mode")

        >>> @skrub.deferred
        ... def mul(value, factor):
        ...     result = value * factor
        ...     print(f"{value} * {factor} = {result}")
        ...     return result

        >>> b = mode.skb.match(
        ...     {"one": mul(a, 1.0), "two": mul(a, 2.0), "three": mul(a, 3.0)},
        ...     default=mul(a, -1.0),
        ... )
        >>> b.skb.eval({"a": 10.0, "mode": "two"})
        10.0 * 2.0 = 20.0
        20.0
        >>> b.skb.eval({"a": 10.0, "mode": "three"})
        10.0 * 3.0 = 30.0
        30.0
        >>> b.skb.eval({"a": 10.0, "mode": "twenty"})
        10.0 * -1.0 = -10.0
        -10.0

        Note that only one of the multiplications gets evaluated.
        """
        return DataOp(Match(self._data_op, targets, default))

    @check_data_op
    def select(self, cols):
        """Select a subset of columns.

        ``cols`` can be a column name or a list of column names, but also a
        skrub selector. Importantly, the exact list of columns that match the
        selector is stored during ``fit`` and then this same list of columns is
        selected during ``transform``.

        Parameters
        ----------
        cols : string, list of strings, or skrub selector
            The columns to select

        Returns
        -------
        dataframe with only the selected columns

        Examples
        --------
        >>> import skrub
        >>> from skrub import selectors as s
        >>> X = skrub.X(skrub.datasets.toy_orders().X)
        >>> X
        <Var 'X'>
        Result:
        ―――――――
           ID product  quantity        date
        0   1     pen         2  2020-04-03
        1   2     cup         3  2020-04-04
        2   3     cup         5  2020-04-04
        3   4   spoon         1  2020-04-05
        >>> X.skb.select(['product', 'quantity'])
        <Apply SelectCols>
        Result:
        ―――――――
          product  quantity
        0     pen         2
        1     cup         3
        2     cup         5
        3   spoon         1
        >>> X.skb.select(s.string())
        <Apply SelectCols>
        Result:
        ―――――――
          product        date
        0     pen  2020-04-03
        1     cup  2020-04-04
        2     cup  2020-04-04
        3   spoon  2020-04-05
        """
        return self._apply(SelectCols(cols), how="no_wrap")

    @check_data_op
    def drop(self, cols):
        """Drop some columns.

        ``cols`` can be a column name or a list of column names, but also a
        skrub selector. Importantly, the exact list of columns that match the
        selector is stored during ``fit`` and then this same list of columns is
        dropped during ``transform``.

        Parameters
        ----------
        cols : string, list of strings, or skrub selector
            The columns to select

        Returns
        -------
        dataframe without the dropped columns

        Examples
        --------
        >>> import skrub
        >>> from skrub import selectors as s
        >>> X = skrub.X(skrub.datasets.toy_orders().X)
        >>> X
        <Var 'X'>
        Result:
        ―――――――
           ID product  quantity        date
        0   1     pen         2  2020-04-03
        1   2     cup         3  2020-04-04
        2   3     cup         5  2020-04-04
        3   4   spoon         1  2020-04-05
        >>> X.skb.drop(['ID', 'date'])
        <Apply DropCols>
        Result:
        ―――――――
          product  quantity
        0     pen         2
        1     cup         3
        2     cup         5
        3   spoon         1
        >>> X.skb.drop(s.string())
        <Apply DropCols>
        Result:
        ―――――――
           ID  quantity
        0   1         2
        1   2         3
        2   3         5
        3   4         1
        """
        return self._apply(DropCols(cols), how="no_wrap")

    @check_data_op
    def concat(self, others, axis=0):
        """Concatenate dataframes vertically or horizontally.

        Parameters
        ----------
        others : list of dataframes
            The dataframes to stack horizontally with ``self``
        axis : {0, 1}, default 0
            The axis to concatenate along.
            0: stack vertically (rows)
            1: stack horizontally (columns)

        Returns
        -------
        dataframe
            The combined dataframes.

        Examples
        --------
        >>> import pandas as pd
        >>> import skrub
        >>> a = skrub.var('a', pd.DataFrame({'a1': [0], 'a2': [1]}))
        >>> b = skrub.var('b', pd.DataFrame({'b1': [2], 'b2': [3]}))
        >>> c = skrub.var('c', pd.DataFrame({'c1': [4], 'c2': [5]}))
        >>> d = skrub.var('d', pd.DataFrame({'c1': [6], 'c2': [7]}))
        >>> a
        <Var 'a'>
        Result:
        ―――――――
           a1  a2
        0   0   1
        >>> a.skb.concat([b, c], axis=1)
        <Concat: 3 dataframes>
        Result:
        ―――――――
           a1  a2  b1  b2  c1  c2
        0   0   1   2   3   4   5

        >>> c.skb.concat([d], axis=0)
        <Concat: 2 dataframes>
        Result:
        ―――――――
           c1  c2
        0   4   5
        1   6   7


        Note that even if we want to concatenate a single dataframe we must
        still put it in a list:

        >>> a.skb.concat([b], axis=1)
        <Concat: 2 dataframes>
        Result:
        ―――――――
           a1  a2  b1  b2
        0   0   1   2   3
        """  # noqa: E501
        return DataOp(Concat(self._data_op, others, axis=axis))

    @check_data_op
    def subsample(self, n=1000, *, how="head"):
        """Configure subsampling of a dataframe or numpy array.

        Enables faster development by computing the previews on a subsample of
        the available data. Outside of previews, no subsampling takes place by
        default but it can be turned on with the ``keep_subsampling`` parameter
        -- see the Notes section for details.

        Parameters
        ----------
        n : int, default=1000
            Number of rows to keep.

        how : 'head' or 'random'
            How subsampling should be done (when it takes place). If 'head',
            the first ``n`` rows are kept. If 'random', ``n`` rows are sampled
            randomly, without maintaining order and without replacement.

        Returns
        -------
        subsampled data
            The subsampled dataframe, column or numpy array.

        See Also
        --------
        DataOp.skb.preview :
            Access a preview of the result on the subsampled data.

        Notes
        -----
        This method configures *how* the dataframe should be subsampled. If it
        has been configured, subsampling actually only takes place in some
        specific situations:

        - When computing the previews (results displayed when printing a
          DataOp and the output of :meth:`DataOp.skb.preview`).
        - When it is explicitly requested by passing ``keep_subsampling=True`` to one
          of the functions that expose that parameter such as
          :meth:`DataOp.skb.make_randomized_search` or :func:`cross_validate`.

        When subsampling has not been configured (``subsample`` has not
        been called anywhere in the DataOp plan), no subsampling is ever done.

        Subsampling is never performed during inference (using the ``predict`` or ``score`` methods), as this would
        lead to inconsistent shapes (number of samples) between the predictions
        and the ground truth labels.

        This method can only be used on steps that produce a dataframe, a
        column (series) or a numpy array.

        Note that subsampling is local to the variable that is being subsampled.
        This means that, if two variables are meant to have the same number of
        rows (e.g., ``X`` and ``y``), both should be subsampled with the same
        strategy.

        The seed for the ``random`` strategy is fixed, so the sampled rows are
        constant when sampling across different variables.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> import skrub

        >>> df = load_diabetes(as_frame=True)["frame"]
        >>> df.shape
        (442, 11)


        >>> data = skrub.var("data", df).skb.subsample(n=15)

        We can see that the previews use only a subsample of 15 rows:

        >>> data.shape
        <GetAttr 'shape'>
        Result (on a subsample):
        ――――――――――――――――――――――――
        (15, 11)
        >>> X = data.drop("target", axis=1, errors="ignore").skb.mark_as_X()
        >>> y = data["target"].skb.mark_as_y()
        >>> pred = X.skb.apply(
        ...     Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α")), y=y
        ... )

        Here also, the preview for the predictions contains 15 rows:

        >>> pred
        <Apply Ridge>
        Result (on a subsample):
        ――――――――――――――――――――――――
        0     142.866906
        1     130.980765
        2     138.555388
        3     149.703363
        4     136.015214
        5     139.773213
        6     134.110415
        7     129.224783
        8     140.161363
        9     155.272033
        10    139.552110
        11    130.318783
        12    135.956591
        13    142.998060
        14    132.511013
        Name: target, dtype: float64

        By default, model fitting and hyperparameter search are done on the
        full data, so if we want the subsampling to take place we have to
        pass ``keep_subsampling=True``:

        >>> quick_search = pred.skb.make_randomized_search(
        ...     keep_subsampling=True, fitted=True, n_iter=4, random_state=0
        ... )
        >>> quick_search.detailed_results_[["mean_test_score", "mean_fit_time", "α"]] # doctest: +SKIP
           mean_test_score  mean_fit_time         α
        0        -0.597596       0.004322  0.431171
        1        -0.599036       0.004328  0.443038
        2        -0.615900       0.004272  0.643117
        3        -0.637498       0.004219  1.398196

        Now that we have checked our learner works on a subsample, we can
        fit the hyperparameter search on the full data:

        >>> full_search = pred.skb.make_randomized_search(
        ...     fitted=True, n_iter=4, random_state=0
        ... )
        >>> full_search.detailed_results_[["mean_test_score", "mean_fit_time", "α"]] # doctest: +SKIP
           mean_test_score  mean_fit_time         α
        0         0.457807       0.004791  0.431171
        1         0.456808       0.004834  0.443038
        2         0.439670       0.004849  0.643117
        3         0.380719       0.004827  1.398196

        This example dataset is so small that the subsampling does not change
        the fit computation time but we can tell the second search used the
        full data from the higher scores. For datasets of a realistic size
        using the subsampling allows us to do a "dry run" of the
        cross-validation or model fitting much faster than when using the
        full data.

        Sampling only one variable does not sample the other:

        >>> data = skrub.var("data", df)
        >>> X = data.drop("target", axis=1, errors="ignore").skb.mark_as_X()
        >>> X = X.skb.subsample(n=15)
        >>> y = data["target"].skb.mark_as_y()
        >>> X.shape
        <GetAttr 'shape'>
        Result (on a subsample):
        ――――――――――――――――――――――――
        (15, 10)
        >>> y.shape
        <GetAttr 'shape'>
        Result:
        ―――――――
        (442,)

        Read more about subsampling in the :ref:`User Guide <user_guide_data_ops_subsampling>`.

        """  # noqa : E501
        return DataOp(SubsamplePreviews(self._data_op, n=n, how=how))

    def clone(self, drop_values=True):
        """Get an independent clone of the DataOp.

        Parameters
        ----------
        drop_values : bool, default=True
            Whether to drop the initial values passed to ``skrub.var()``.
            This is convenient for example to serialize DataOps without
            creating large files.

        Returns
        -------
        clone
            A new DataOp which does not share its state (such as fitted
            estimators) or cache with the original, and possibly without the
            variables' values.

        Examples
        --------
        >>> import skrub
        >>> c = skrub.var('a', 0) + skrub.var('b', 1)
        >>> c
        <BinOp: add>
        Result:
        ―――――――
        1
        >>> c.skb.get_data()
        {'a': 0, 'b': 1}
        >>> clone = c.skb.clone()
        >>> clone
        <BinOp: add>
        >>> clone.skb.get_data()
        {}

        We can ask to keep the variable values:

        >>> clone = c.skb.clone(drop_values=False)
        >>> clone.skb.get_data()
        {'a': 0, 'b': 1}

        Note that in that case the cache used for previews is still cleared. So
        if we want the preview we need to prime the new DataOp by
        accessing the preview once (either directly or by adding more steps to it):

        >>> clone
        <BinOp: add>
        >>> clone.skb.preview()
        1
        >>> clone
        <BinOp: add>
        Result:
        ―――――――
        1
        """

        return clone(self._data_op, drop_preview_data=drop_values)

    def eval(self, environment=None, *, keep_subsampling=False):
        """Evaluate the DataOp.

        This returns the result produced by evaluating the DataOp, ie
        running the corresponding learner. The result is always the output
        of the learner's ``fit_transform`` -- a learner is refitted to the
        provided data.

        If no data is provided, the values passed when creating the variables
        in the DataOp are used.

        Parameters
        ----------
        environment : dict or None, optional
            If ``None``, the initial values of the variables contained in the
            DataOp are used. If a dict, it must map the name of each
            variable to a corresponding value.

        keep_subsampling : bool, default=False
            If True, and if subsampling has been configured (see
            :meth:`DataOp.skb.subsample`), use a subsample of the data. By
            default subsampling is not applied and all the data is used.

        Returns
        -------
        result
            The result of running the computation, ie of executing the
            learner's ``fit_transform`` on the provided data.

        See Also
        --------
        DataOp.skb.preview :
            Access the preview of the result on the variables initial values,
            with subsampling. Faster than ``eval`` but does not allow passing
            new data and always applies subsampling.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var('a', 10)
        >>> b = skrub.var('b', 5)
        >>> c = a + b
        >>> c
        <BinOp: add>
        Result:
        ―――――――
        15
        >>> c.skb.eval()
        15
        >>> c.skb.eval({'a': 1, 'b': 2})
        3
        """
        if environment is not None and not isinstance(environment, typing.Mapping):
            raise TypeError(
                "The `environment` passed to `eval()` should be None or a dictionary, "
                f"got: '{type(environment)}'"
            )
        if environment is None:
            environment = self.get_data()
        else:
            environment = {
                **environment,
                "_skrub_use_var_values": not _var_values_provided(
                    self._data_op, environment
                ),
            }
        environment = env_with_subsampling(self._data_op, environment, keep_subsampling)
        return evaluate(
            self._data_op, mode="fit_transform", environment=environment, clear=True
        )

    def preview(self):
        """Get the value computed for previews (shown when printing the DataOp).

        Returns
        -------
        preview result
            The result of evaluating the DataOp on the data stored in its
            variables.

        See Also
        --------
        DataOp.skb.subsample :
            Specify how to subsample an intermediate result when computing
            previews.

        DataOp.skb.eval :
            Evaluate the DataOp. Unlike ``preview``, we can pass new data
            rather than using the values that variables were initialized with,
            but results are not cached, and no subsampling takes place by
            default.

        Examples
        --------
        >>> import skrub

        >>> a = skrub.var('a', 1)
        >>> b = skrub.var('b', 2)
        >>> c = a + b

        When we display a DataOp, we see a preview of the result (``3`` in
        this case):

        >>> c
        <BinOp: add>
        Result:
        ―――――――
        3

        If we want to actually access that value ``3``, rather than just seeing
        it displayed, we can use ``.skb.preview()``:

        >>> c.skb.preview()
        3

        This is the actual number ``3``, not a DataOp or just a display

        >>> type(c.skb.preview())
        <class 'int'>

        Accessing the preview is usually faster than calling ``.skb.eval()``
        because results are cached and subsampling is used by default.
        """
        return evaluate(self._data_op, mode="preview", environment=None, clear=False)

    @check_data_op
    def freeze_after_fit(self):
        """Freeze the result during learner fitting.

        With ``freeze_after_fit()`` the result of the DataOp is
        computed during ``fit()``, and then reused (not recomputed) during
        ``transform()`` or ``predict()``.

        Returns
        -------
        The DataOp whose value does not change after ``fit()``

        Notes
        -----
        This is an advanced functionality, and the need for it is usually
        an indication that we need to define a custom scikit-learn transformer
        that we can use with ``.skb.apply()``.

        Examples
        --------
        >>> import skrub
        >>> X_df = skrub.datasets.toy_orders().X
        >>> X_df
           ID product  quantity        date
        0   1     pen         2  2020-04-03
        1   2     cup         3  2020-04-04
        2   3     cup         5  2020-04-04
        3   4   spoon         1  2020-04-05
        >>> n_products = skrub.X()['product'].nunique()
        >>> transformer = n_products.skb.make_learner()
        >>> transformer.fit_transform({'X': X_df})
        3

        If we take only the first 2 rows ``nunique()`` (a stateless function)
        returns ``2``:

        >>> transformer.transform({'X': X_df.iloc[:2]})
        2

        If instead of recomputing it we want the number of products to be
        remembered during ``fit`` and reused during ``transform``:

        >>> n_products = skrub.X()['product'].nunique().skb.freeze_after_fit()
        >>> transformer = n_products.skb.make_learner()
        >>> transformer.fit_transform({'X': X_df})
        3
        >>> transformer.transform({'X': X_df.iloc[:2]})
        3
        """
        return DataOp(FreezeAfterFit(self._data_op))

    def get_data(self):
        """Collect the values of the variables contained in the DataOp.

        Returns
        -------
        dict mapping variable names to their values
            Variables for which no value was given do not appear in the result.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var('a', 0)
        >>> b = skrub.var('b', 1)
        >>> c = skrub.var('c') # note no value
        >>> e = a + b + c
        >>> e.skb.get_data()
        {'a': 0, 'b': 1}
        """

        data = {}

        for n in nodes(self._data_op):
            impl = n._skrub_impl
            if isinstance(impl, Var) and impl.value is not NULL:
                data[impl.name] = impl.value
        return data

    def get_vars(self, all_named_ops=False):
        """
        Get all the variables used in the DataOp.

        Parameters
        ----------
        all_named_ops : bool, default = False
            If False, return only actual variables (DataOps created with
            :func:`var()`, :func:`X()` or :func:`y()`). If False, return all
            nodes that have a name (ie for which a value can be passed in the
            environment).

        Returns
        -------
        dict :
            Keys are names, and values the corresponding DataOp.

        Examples
        --------
        >>> import skrub

        >>> a = skrub.var("a")
        >>> b = skrub.var("b")
        >>> c = (a + b).skb.set_name("c")
        >>> d = c + c
        >>> d
        <BinOp: add>

        Our DataOp, `d`, contains 2 variables: "a" and "b":

        >>> d.skb.get_vars()
        {'a': <Var 'a'>, 'b': <Var 'b'>}

        Those are the keys for which we need to provide values in the
        environment when evaluating `d`:

        >>> d.skb.eval({"a": 10, "b": 3}) # (10 + 3) + (10 + 3) = 26
        26

        In addition, we set a name on the internal node `c`. It is not a
        variable, and normally it is computed as `(a + b)`. But as it has a
        name, we can override its output by passing a value for "c" in the
        environment. When we do, the computation of `c` never happens (nor of
        `a` or `b`, here, because they are only used to compute `c`) -- it is
        bypassed and the provided value is used instead.

        >>> d.skb.eval({"c": 7}) # 7 + 7 = 14
        14

        If we want ``get_vars`` to also list nodes like our example ``c`` which
        have a name and can be passed in the environment, we pass
        ``all_named_ops=True``:

        >>> d.skb.get_vars(all_named_ops=True)
        {'a': <Var 'a'>, 'b': <Var 'b'>, 'c': <c | BinOp: add>}

        Note ``get_vars`` can be particularly useful when we have a learner
        (e.g. loaded from a pickle file) and we want to check what inputs we
        should pass to its methods such as ``fit`` and ``transform``:

        >>> learner = d.skb.make_learner()
        >>> list(learner.data_op.skb.get_vars().keys())
        ['a', 'b']

        The output above tells us what keys the dict we pass to
        ``learner.fit()`` should contain:

        >>> learner.fit({'a': 2, 'b': 3})
        SkrubLearner(data_op=<BinOp: add>)
        """
        from ._data_ops import Var
        from ._evaluation import nodes

        named_nodes = {
            name: op
            for op in nodes(self._data_op)
            if (name := op._skrub_impl.name) is not None
        }
        if all_named_ops:
            return named_nodes
        return {
            name: op
            for name, op in named_nodes.items()
            if isinstance(op._skrub_impl, Var)
        }

    def draw_graph(self):
        """Get an SVG string representing the computation graph.

        In addition to the usual ``str`` methods, the result has an ``open()``
        method which displays it in a web browser window.

        Returns
        -------
        GraphDrawing
           Drawing of the computation graph. This objects has attributes
           ``svg`` and ``png``, containing representations of the graph in
           those formats (as ``bytes`` objects), and a method ``.open()`` to
           display it in a browser window.
        """

        return draw_data_op_graph(self._data_op)

    def describe_steps(self):
        """Get a text representation of the computation graph.

        Usually the graphical representation provided by :meth:`DataOp.skb.draw_graph`
        or :meth:`DataOp.skb.full_report` is more useful. This is a fallback for
        inspecting the computation graph when only text output is available.

        Returns
        -------
        str
            A string representing the different computation steps, one on each
            line.

        See Also
        --------
        :func:`sklearn.model_selection.cross_validate`:
            Evaluate metric(s) by cross-validation and also record fit/score times.

        :func:`skrub.DataOp.skb.make_learner`:
            Get a skrub learner for this DataOp.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var('a')
        >>> b = skrub.var('b')
        >>> c = a + b
        >>> d = c * c
        >>> print(d.skb.describe_steps())
        Var 'a'
        Var 'b'
        BinOp: add
        ( Var 'a' )*
        ( Var 'b' )*
        ( BinOp: add )*
        BinOp: mul
        * Cached, not recomputed

        The above should be read from top to bottom as instructions for a
        simple stack machine: load the variable 'a', load the variable 'b',
        compute the addition leaving the result of (a + b) on the stack, then
        repeat this operation (but the second time no computation actually runs
        because the result of evaluating ``c`` has been cached in-memory), and
        finally evaluate the multiplication.
        """

        return describe_steps(self._data_op)

    def describe_param_grid(self):
        """Describe the hyper-parameters extracted from choices in the DataOp.

        DataOps can contain choices, ranges of possible values to be tuned
        by hyperparameter search. This function provides a description of the
        grid (set of combinations) of hyperparameters extracted from the
        DataOp.

        Please refer to the examples gallery for a full explanation of choices
        and hyper-parameter tuning.

        Returns
        -------
        str
            A textual description of the different choices contained in this
            DataOp.

        Examples
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.feature_selection import SelectKBest

        >>> import skrub

        >>> X = skrub.X()
        >>> y = skrub.y()

        >>> dim_reduction = skrub.choose_from(
        ...     {
        ...         "PCA": PCA(
        ...             n_components=skrub.choose_int(
        ...                 5, 100, log=True, name="n_components"
        ...             )
        ...         ),
        ...         "SelectKBest": SelectKBest(
        ...             k=skrub.choose_int(5, 100, log=True, name="k")
        ...         ),
        ...     },
        ...     name="dim_reduction",
        ... )
        >>> selected = X.skb.apply(dim_reduction)
        >>> classifier = skrub.choose_from(
        ...     {
        ...         "logreg": LogisticRegression(
        ...             C=skrub.choose_float(0.001, 100, log=True, name="C")
        ...         ),
        ...         "rf": RandomForestClassifier(
        ...             n_estimators=skrub.choose_int(20, 400, name="N 🌴")
        ...         ),
        ...     },
        ...     name="classifier",
        ... )
        >>> pred = selected.skb.apply(classifier, y=y)
        >>> print(pred.skb.describe_param_grid())
        - dim_reduction: 'PCA'
          n_components: choose_int(5, 100, log=True, name='n_components')
          classifier: 'logreg'
          C: choose_float(0.001, 100, log=True, name='C')
        - dim_reduction: 'PCA'
          n_components: choose_int(5, 100, log=True, name='n_components')
          classifier: 'rf'
          N 🌴: choose_int(20, 400, name='N 🌴')
        - dim_reduction: 'SelectKBest'
          k: choose_int(5, 100, log=True, name='k')
          classifier: 'logreg'
          C: choose_float(0.001, 100, log=True, name='C')
        - dim_reduction: 'SelectKBest'
          k: choose_int(5, 100, log=True, name='k')
          classifier: 'rf'
          N 🌴: choose_int(20, 400, name='N 🌴')

        Sampling a configuration for this learner starts by selecting an entry
        (marked by ``-``) in the list above, then a value for each of the
        hyperparameters listed (used) in that entry. For example note that the
        configurations that use the random forest do not list the
        hyperparameter ``C`` which is used only by the logistic regression.
        """

        return describe_param_grid(self._data_op)

    def describe_defaults(self):
        """Describe the hyper-parameters used by the default learner.

        Returns a dict mapping choice names to a simplified representation of
        the corresponding value in the default learner.

        Examples
        --------
        >>> import skrub
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.feature_selection import SelectKBest
        >>> from sklearn.ensemble import RandomForestClassifier

        >>> X, y = skrub.X(), skrub.y()
        >>> selector = SelectKBest(k=skrub.choose_int(4, 20, log=True, name='k'))
        >>> logistic = LogisticRegression(
        ...     C=skrub.choose_float(0.1, 10.0, log=True, name="C"),
        ... )
        >>> rf = RandomForestClassifier(
        ...     n_estimators=skrub.choose_int(3, 30, log=True, name="N 🌴"),
        ...     random_state=0,
        ... )
        >>> classifier = skrub.choose_from(
        ...     {"logistic": logistic, "rf": rf}, name="classifier"
        ... )
        >>> pred = X.skb.apply(selector, y=y).skb.apply(classifier, y=y)
        >>> print(pred.skb.describe_defaults())
        {'k': 9, 'classifier': 'logistic', 'C': 1.0...}
        """
        from ._evaluation import choice_graph, chosen_or_default_outcomes
        from ._inspection import describe_params

        return describe_params(
            chosen_or_default_outcomes(self._data_op), choice_graph(self._data_op)
        )

    def full_report(
        self,
        environment=None,
        open=True,
        output_dir=None,
        overwrite=False,
        title=None,
    ):
        """Generate a full report of the DataOp's evaluation.

        This creates a report showing the computation graph, and for each
        intermediate computation, some information (the line of code where it
        was defined, the time it took to run, and more) and a display of the
        intermediate result (or error). By default, the report is stored in
        a timestamped subdirectory of the skrub data folder.

        .. note::
            When this function is invoked reports starting with ``full_data_op_report_``
            that are stored in the skrub data folder are automatically deleted after 7 days.
            This is to avoid accumulating too many reports over time. If you want to keep
            specific reports, please specify an output directory.


        Parameters
        ----------
        environment : dict or None (default=None)
            Bindings for variables and choices contained in the DataOp. If
            not provided, the variables' ``value`` and the choices default
            value are used.

        open : bool (default=True)
            Whether to open the report in a web browser once computed.

        output_dir : str or Path or None (default=None)
            Directory where to store the report. If ``None``, a timestamped
            subdirectory will be created in the skrub data directory. Note
            that the reports created with ``output_dir=None`` are automatically
            deleted after 7 days.

        overwrite : bool (default=False)
            What to do if the output directory already exists. If
            ``overwrite``, replace it, otherwise raise an exception.

        title: str (default=None)
            Title to display at the top of the report. If ``None``, no title will be
            displayed.

        Returns
        -------
        dict
            The results of evaluating the DataOp. The keys are
            ``'result'``, ``'error'`` and ``'report_path'``. If the execution
            raised an exception, it is contained in ``'error'`` and
            ``'result'`` is ``None``. Otherwise the result produced by the
            evaluation is in ``'result'`` and ``'error'`` is ``None``. Either
            way a report is stored at the location indicated by
            ``'report_path'``.

        See Also
        --------
        :meth:`SkrubLearner.report` :
            Generate a report for a call to any of the methods of the
            :class:`SkrubLearner` such as ``transform()``, ``predict()``,
            ``predict_proba()`` etc.

        Notes
        -----
        The learner is run doing a ``fit_transform``. To get a report for other
        methods (e.g. ``predict``, see :meth:`SkrubLearner.report`). If
        ``environment`` is provided, it is used as the bindings for the
        variables in the DataOp, and otherwise, the ``value`` attributes of the
        variables are used.

        At the moment, this creates a directory on the filesystem containing
        HTML files. The report can be displayed by visiting the contained
        ``index.html`` in a webbrowser, or passing ``open=True`` (the default)
        to this method.

        Examples
        --------
        >>> # ignore this line:
        >>> import pytest; pytest.skip('graphviz may not be installed')

        >>> import skrub
        >>> c = skrub.var('a', 1) / skrub.var('b', 2)
        >>> report = c.skb.full_report(open=False)
        >>> report['result']
        0.5
        >>> report['error']
        >>> report['report_path']
        PosixPath('.../skrub_data/execution_reports/full_data_op_report_.../index.html')

        We pass data:

        >>> report = c.skb.full_report({'a': 33, 'b': 11 }, open=False)
        >>> report['result']
        3.0

        And if there was an error:

        >>> report = c.skb.full_report({'a': 1, 'b': 0}, open=False)
        >>> report['result']
        >>> report['error']
        ZeroDivisionError('division by zero')
        >>> report['report_path']
        PosixPath('.../skrub_data/execution_reports/full_data_op_report_.../index.html')
        """  # noqa : E501

        if environment is None:
            mode = "preview"
            clear = False
        else:
            mode = "fit_transform"
            clear = True

        return full_report(
            self._data_op,
            environment=environment,
            mode=mode,
            clear=clear,
            open=open,
            output_dir=output_dir,
            overwrite=overwrite,
            title=title,
        )

    def make_learner(self, *, fitted=False, keep_subsampling=False):
        """Get a skrub learner for this DataOp.

        Returns a :class:`SkrubLearner` with a ``fit()`` method so it can be fit
        to some training data and then apply it to unseen data by calling
        ``transform()`` or ``predict()``. Unlike scikit-learn estimators, skrub
        learners accept a dictionary of inputs rather than ``X`` and ``y`` arguments.

        .. warning::

           If the DataOp contains choices (e.g. ``choose_from(...)``), this
           learner uses the default value of each choice. To actually pick the
           best value with hyperparameter tuning, use
           :meth:`DataOp.skb.make_randomized_search` or
           :meth:`DataOp.skb.make_grid_search` instead.

        Parameters
        ----------
        fitted : bool (default=False)
            If true, the returned learner is fitted to the data provided when
            initializing variables in ``skrub.var("name", value=...)`` and
            ``skrub.X(value)``.

        keep_subsampling : bool (default=False)
            If True, and if subsampling has been configured (see
            :meth:`DataOp.skb.subsample`), fit on a subsample of the data. By
            default subsampling is not applied and all the data is used. This
            is only applied for fitting the estimator when ``fitted=True``,
            subsequent use of the estimator is not affected by subsampling.
            Therefore it is an error to pass ``keep_subsampling=True`` and
            ``fitted=False`` (because ``keep_subsampling=True`` would have no
            effect).

        Returns
        -------
        learner
            A skrub learner with an interface similar to scikit-learn's, except
            that its methods accept a dictionary of named inputs rather than
            ``X`` and ``y`` arguments.

        Examples
        --------
        >>> import skrub
        >>> from sklearn.dummy import DummyClassifier
        >>> orders_df = skrub.datasets.toy_orders(split="train").orders
        >>> orders = skrub.var('orders', orders_df)
        >>> X = orders.drop(columns='delayed', errors='ignore').skb.mark_as_X()
        >>> y = orders['delayed'].skb.mark_as_y()
        >>> pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(
        ...     DummyClassifier(), y=y
        ... )
        >>> pred
        <Apply DummyClassifier>
        Result:
        ―――――――
        0    False
        1    False
        2    False
        3    False
        Name: delayed, dtype: bool
        >>> learner = pred.skb.make_learner(fitted=True)
        >>> new_orders_df = skrub.datasets.toy_orders(split='test').X
        >>> new_orders_df
           ID product  quantity        date
        4   5     cup         5  2020-04-11
        5   6    fork         2  2020-04-12
        >>> learner.predict({'orders': new_orders_df})
        array([False, False])

        Note that the ``'orders'`` key in the dictionary passed to ``predict``
        corresponds to the name ``'orders'`` in ``skrub.var('orders',
        orders_df)`` above.

        Please see the examples gallery for full information about DataOps
        and the learners they generate.
        """
        _check_keep_subsampling(fitted, keep_subsampling)

        learner = SkrubLearner(self.clone())
        _check_can_be_pickled(learner)
        if not fitted:
            return learner
        return learner.fit(
            env_with_subsampling(self._data_op, self.get_data(), keep_subsampling)
        )

    def train_test_split(
        self,
        environment=None,
        *,
        keep_subsampling=False,
        split_func=model_selection.train_test_split,
        **split_func_kwargs,
    ):
        """Split an environment into a training an testing environments.

        Parameters
        ----------
        environment : dict, optional
            The environment (dict mapping variable names to values) containing the
            full data. If ``None`` (the default), the data is retrieved from the
            DataOp.

        keep_subsampling : bool, default=False
            If True, and if subsampling has been configured (see
            :meth:`DataOp.skb.subsample`), use a subsample of the data. By
            default subsampling is not applied and all the data is used.

        split_func : function, optional
            The function used to split X and y once they have been computed. By
            default, :func:`~sklearn.model_selection.train_test_split` is used.

        split_func_kwargs
            Additional named arguments to pass to the splitting function.

        Returns
        -------
        dict
            The return value is slightly different than scikit-learn's. Rather than
            a tuple, it returns a dictionary with the following keys:

            - train: a dictionary containing the training environment
            - test: a dictionary containing the test environment
            - X_train: the value of the variable marked with ``skb.mark_as_X()`` in
              the train environment
            - X_test: the value of the variable marked with ``skb.mark_as_X()`` in
              the test environment
            - y_train: the value of the variable marked with ``skb.mark_as_y()`` in
              the train environment, if there is one (may not be the case for
              unsupervised learning).
            - y_test: the value of the variable marked with ``skb.mark_as_y()`` in
              the test environment, if there is one (may not be the case for
              unsupervised learning).

        Examples
        --------
        >>> import skrub
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.metrics import accuracy_score

        >>> orders = skrub.var("orders", skrub.datasets.toy_orders().orders)
        >>> X = orders.skb.drop("delayed").skb.mark_as_X()
        >>> y = orders["delayed"].skb.mark_as_y()
        >>> delayed = X.skb.apply(skrub.TableVectorizer()).skb.apply(
        ...     DummyClassifier(), y=y
        ... )

        >>> split = delayed.skb.train_test_split(random_state=0)
        >>> split.keys()
        dict_keys(['train', 'test', 'X_train', 'X_test', 'y_train', 'y_test'])
        >>> learner = delayed.skb.make_learner()
        >>> learner.fit(split["train"])
        SkrubLearner(data_op=<Apply DummyClassifier>)
        >>> learner.score(split["test"])
        0.0
        >>> predictions = learner.predict(split["test"])
        >>> accuracy_score(split["y_test"], predictions)
        0.0
        """
        if (splitter := split_func_kwargs.pop("splitter", None)) is not None:
            warnings.warn(
                (
                    "The `splitter` parameter of `.skb.train_test_split` has been"
                    " renamed `split_func`. Using it will raise an error in a future"
                    " release of skrub."
                ),
                category=FutureWarning,
            )
            split_func = splitter
        if environment is None:
            environment = self.get_data()
        return train_test_split(
            self._data_op,
            environment,
            keep_subsampling=keep_subsampling,
            split_func=split_func,
            **split_func_kwargs,
        )

    def iter_cv_splits(self, environment=None, *, keep_subsampling=False, cv=KFOLD_5):
        """Yield splits of an environment into training and testing environments.

        Parameters
        ----------
        environment : dict, optional
            The environment (dict mapping variable names to values) containing the
            full data. If ``None`` (the default), the data is retrieved from the
            DataOp.

        keep_subsampling : bool, default=False
            If True, and if subsampling has been configured (see
            :meth:`DataOp.skb.subsample`), use a subsample of the data. By
            default subsampling is not applied and all the data is used.

        cv : int, cross-validation generator or iterable, default=KFold(5)
            The default is 5-fold without shuffling. Can be a cross-validation
            splitter, an iterable yielding pairs of (train, test) indices, or an
            int to specify the number of folds for KFold splitting.

        Yields
        ------
        dict
            For each split, a dict is produced, containing the following keys:

            - train: a dictionary containing the training environment
            - test: a dictionary containing the test environment
            - X_train: the value of the variable marked with ``skb.mark_as_X()`` in
              the train environment
            - X_test: the value of the variable marked with ``skb.mark_as_X()`` in
              the test environment
            - y_train: the value of the variable marked with ``skb.mark_as_y()`` in
              the train environment, if there is one (may not be the case for
              unsupervised learning).
            - y_test: the value of the variable marked with ``skb.mark_as_y()`` in
              the test environment, if there is one (may not be the case for
              unsupervised learning).

        Examples
        --------
        >>> import skrub
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.metrics import accuracy_score

        >>> orders = skrub.var("orders")
        >>> X = orders.skb.drop("delayed").skb.mark_as_X()
        >>> y = orders["delayed"].skb.mark_as_y()
        >>> delayed = X.skb.apply(skrub.TableVectorizer()).skb.apply(
        ...     DummyClassifier(), y=y
        ... )
        >>> df = skrub.datasets.toy_orders().orders
        >>> accuracies = []
        >>> for split in delayed.skb.iter_cv_splits({"orders": df}, cv=3):
        ...     learner = delayed.skb.make_learner().fit(split["train"])
        ...     prediction = learner.predict(split["test"])
        ...     accuracies.append(accuracy_score(split["y_test"], prediction))
        >>> accuracies
        [1.0, 0.0, 1.0]
        """
        if environment is None:
            environment = self.get_data()
        yield from iter_cv_splits(
            self._data_op, environment, keep_subsampling=keep_subsampling, cv=cv
        )

    def make_grid_search(self, *, fitted=False, keep_subsampling=False, **kwargs):
        """Find the best parameters with grid search.

        This function returns a :class:`ParamSearch`, an object similar to
        scikit-learn's :class:`~sklearn.model_selection.RandomizedSearchCV`, where the main difference is that
        ``fit()`` and ``predict()`` accept a dictionary of inputs
        rather than ``X`` and ``y``. The best learner can
        be returned by calling ``.best_learner_``.

        Parameters
        ----------
        fitted : bool (default=False)
            If ``True``, the gridsearch is fitted on the data provided when
            initializing variables in this DataOp (the data returned by
            ``.skb.get_data()``).

        keep_subsampling : bool (default=False)
            If True, and if subsampling has been configured (see
            :meth:`DataOp.skb.subsample`), fit on a subsample of the data. By
            default subsampling is not applied and all the data is used. This
            is only applied for fitting the grid search when ``fitted=True``,
            subsequent use of the grid search is not affected by subsampling.
            Therefore it is an error to pass ``keep_subsampling=True`` and
            ``fitted=False`` (because ``keep_subsampling=True`` would have no
            effect).

        kwargs : dict
            All other named arguments are forwarded to
            ``sklearn.search.GridSearchCV``.

        Returns
        -------
        ParamSearch
            An object implementing the hyperparameter search. Besides the usual
            ``fit``, ``predict``, attributes of interest are
        ``results_``, ``plot_results()``, and ``best_learner_``.

        See also
        --------
        skrub.DataOp.skb.make_randomized_search :
            Find the best parameters with grid search.

        Examples
        --------
        >>> import skrub
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.dummy import DummyClassifier

        >>> X_a, y_a = make_classification(random_state=0)
        >>> X, y = skrub.X(X_a), skrub.y(y_a)
        >>> logistic = LogisticRegression(C=skrub.choose_from([0.1, 10.0], name="C"))
        >>> rf = RandomForestClassifier(
        ...     n_estimators=skrub.choose_from([3, 30], name="N 🌴"),
        ...     random_state=0,
        ... )
        >>> classifier = skrub.choose_from(
        ...     {"logistic": logistic, "rf": rf, "dummy": DummyClassifier()}, name="classifier"
        ... )
        >>> pred = X.skb.apply(classifier, y=y)
        >>> print(pred.skb.describe_param_grid())
        - classifier: 'logistic'
          C: [0.1, 10.0]
        - classifier: 'rf'
          N 🌴: [3, 30]
        - classifier: 'dummy'

        >>> search = pred.skb.make_grid_search(fitted=True)
        >>> search.results_
              C   N 🌴 classifier mean_test_score
        0   NaN  30.0         rf             0.89
        1   0.1   NaN   logistic             0.84
        2  10.0   NaN   logistic             0.80
        3   NaN   3.0         rf             0.65
        4   NaN   NaN      dummy             0.50

        If the DataOp contains some numeric ranges (``choose_float``,
        ``choose_int``), either discretize them by providing the ``n_steps``
        argument or use ``make_randomized_search`` instead of
        ``make_grid_search``.

        >>> logistic = LogisticRegression(
        ...     C=skrub.choose_float(0.1, 10.0, log=True, n_steps=5, name="C")
        ... )
        >>> pred = X.skb.apply(logistic, y=y)
        >>> print(pred.skb.describe_param_grid())
        - C: choose_float(0.1, 10.0, log=True, n_steps=5, name='C')
        >>> search = pred.skb.make_grid_search(fitted=True)
        >>> search.results_
            C	mean_test_score
        0	0.100000	0.84
        1	0.316228	0.83
        2	1.000000	0.81
        3	3.162278	0.80
        4	10.000000	0.80

        Please refer to the examples gallery for an in-depth explanation.
        """  # noqa: E501
        _check_keep_subsampling(fitted, keep_subsampling)
        _check_grid_search_possible(self._data_op)

        search = ParamSearch(
            self.clone(), model_selection.GridSearchCV(None, None, **kwargs)
        )
        if not fitted:
            return search
        return search.fit(
            env_with_subsampling(self._data_op, self.get_data(), keep_subsampling)
        )

    def make_randomized_search(self, *, fitted=False, keep_subsampling=False, **kwargs):
        """Find the best parameters with randomized search.

        This function returns a :class:`ParamSearch`, an object similar to
        scikit-learn's :class:`~sklearn.model_selection.RandomizedSearchCV`, where
        the main difference is ``fit()`` and ``predict()`` accept a
        dictionary of inputs rather than ``X`` and ``y``. The best learner can
        be returned by calling ``.best_learner_``.

        Parameters
        ----------
        fitted : bool (default=False)
            If ``True``, the randomized search is fitted on the data provided when
            initializing variables in this DataOp (the data returned by
            ``.skb.get_data()``).

        keep_subsampling : bool (default=False)
            If True, and if subsampling has been configured (see
            :meth:`DataOp.skb.subsample`), fit on a subsample of the data. By
            default subsampling is not applied and all the data is used. This
            is only applied for fitting the randomized search when ``fitted=True``,
            subsequent use of the randomized search is not affected by subsampling.
            Therefore it is an error to pass ``keep_subsampling=True`` and
            ``fitted=False`` (because ``keep_subsampling=True`` would have no
            effect).

        kwargs : dict
            All other named arguments are forwarded to
            :class:`~sklearn.search.RandomizedSearchCV`.

        Returns
        -------
        ParamSearch
            An object implementing the hyperparameter search. Besides the usual
            ``fit``, ``predict``, attributes of interest are
            ``results_``, ``plot_results()``, and ``best_learner_``.

        See also
        --------
        skrub.DataOp.skb.make_grid_search :
            Find the best parameters with grid search.

        Examples
        --------
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
        ...     n_estimators=skrub.choose_int(3, 30, log=True, name="N 🌴"),
        ...     random_state=0,
        ... )
        >>> classifier = skrub.choose_from(
        ...     {"logistic": logistic, "rf": rf, "dummy": DummyClassifier()}, name="classifier"
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

        >>> search = pred.skb.make_randomized_search(fitted=True, random_state=0)
        >>> search.results_
            k         C  N 🌴 classifier  mean_test_score
        0   4  4.626363  NaN   logistic             0.92
        1  16       NaN  6.0         rf             0.90
        2  11       NaN  7.0         rf             0.88
        3   7  3.832217  NaN   logistic             0.87
        4  10  4.881255  NaN   logistic             0.85
        5  20  3.965675  NaN   logistic             0.80
        6  14       NaN  3.0         rf             0.77
        7   4       NaN  NaN      dummy             0.50
        8  10       NaN  NaN      dummy             0.50
        9   5       NaN  NaN      dummy             0.50

        Please refer to the examples gallery for an in-depth explanation.
        """  # noqa: E501
        _check_keep_subsampling(fitted, keep_subsampling)

        search = ParamSearch(
            self.clone(), model_selection.RandomizedSearchCV(None, None, **kwargs)
        )
        if not fitted:
            return search
        return search.fit(
            env_with_subsampling(self._data_op, self.get_data(), keep_subsampling)
        )

    def iter_learners_grid(self):
        """Get learners with different parameter combinations.

        This generator yields a :class:`SkrubLearner` parametrized for each
        possible combination of choices.

        The choice outcomes used in each learner can be inspected with
        :meth:`SkrubLearner.describe_params()`.

        See Also
        --------
        DataOp.skb.iter_learners_randomized :
            Similar function but for random sampling of the parameter space.
            Must be used when the DataOp contains some numeric ranges built
            with :func:`choose_float` or :func:`choose_int` with
            ``n_steps=None``.

        DataOp.skb.make_grid_search :
            Learner with built-in exhaustive exploration of the parameter grid
            to select the best one.

        DataOp.skb.make_randomized_search :
            Learner with built-in randomized exploration of the parameter grid
            to select the best one.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn import preprocessing
        >>> import skrub

        >>> scaler = skrub.choose_from(
        ...     [
        ...         preprocessing.MinMaxScaler(),
        ...         preprocessing.StandardScaler(),
        ...         preprocessing.RobustScaler(),
        ...         preprocessing.MaxAbsScaler(),
        ...     ],
        ...     name="scaler",
        ... )
        >>> out = skrub.X().skb.apply(scaler)

        >>> X = np.asarray([-4.0, 3.0, 10.0])[:, None]

        >>> for p in out.skb.iter_learners_grid():
        ...     print("======================================")
        ...     print("params:", p.describe_params())
        ...     print("result:")
        ...     print(p.fit_transform({"X": X}))
        ======================================
        params: {'scaler': 'MinMaxScaler()'}
        result:
        [[0. ]
         [0.5]
         [1. ]]
        ======================================
        params: {'scaler': 'StandardScaler()'}
        result:
        [[-1.22474487]
         [ 0.        ]
         [ 1.22474487]]
        ======================================
        params: {'scaler': 'RobustScaler()'}
        result:
        [[-1.]
         [ 0.]
         [ 1.]]
        ======================================
        params: {'scaler': 'MaxAbsScaler()'}
        result:
        [[-0.4]
         [ 0.3]
         [ 1. ]]
        """
        learner = self.make_learner()
        grid = model_selection.ParameterGrid(learner.get_param_grid())
        for params in grid:
            new = self.make_learner()
            new.set_params(**params)
            yield new

    def iter_learners_randomized(self, n_iter, *, random_state=None):
        """Get learners with different parameter combinations.

        This generator yields a :class:`SkrubLearner` parametrized for each
        possible combination of choices.

        The choice outcomes used in each learner can be inspected with
        :meth:`SkrubLearner.describe_params()`.

        See Also
        --------
        DataOp.skb.iter_learners_grid :
            Similar function but for exploring all the possible parameter
            combinations. Cannot be used when the DataOp contains some
            numeric ranges built with :func:`choose_float` or
            :func:`choose_int` with ``n_steps=None``.

        DataOp.skb.make_grid_search :
            Learner with built-in exhaustive exploration of the parameter grid
            to select the best one.

        DataOp.skb.make_randomized_search :
            Learner with built-in randomized exploration of the parameter grid
            to select the best one.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn import preprocessing
        >>> import skrub

        >>> scaler = skrub.choose_from(
        ...     [
        ...         preprocessing.MinMaxScaler(),
        ...         preprocessing.StandardScaler(),
        ...         preprocessing.RobustScaler(),
        ...         preprocessing.MaxAbsScaler(),
        ...     ],
        ...     name="scaler",
        ... )
        >>> out = skrub.X().skb.apply(scaler)

        >>> X = np.asarray([-4.0, 3.0, 10.0])[:, None]

        >>> for p in out.skb.iter_learners_randomized(n_iter=2, random_state=0):
        ...     print("======================================")
        ...     print("params:", p.describe_params())
        ...     print("result:")
        ...     print(p.fit_transform({"X": X}))
        ======================================
        params: {'scaler': 'RobustScaler()'}
        result:
        [[-1.]
         [ 0.]
         [ 1.]]
        ======================================
        params: {'scaler': 'MaxAbsScaler()'}
        result:
        [[-0.4]
         [ 0.3]
         [ 1. ]]
        """
        learner = self.make_learner()
        sampler = model_selection.ParameterSampler(
            learner.get_param_grid(), n_iter=n_iter, random_state=random_state
        )
        for params in sampler:
            new = self.make_learner()
            new.set_params(**params)
            yield new

    def cross_validate(self, environment=None, *, keep_subsampling=False, **kwargs):
        """Cross-validate the DataOp plan.

        This generates the learner with default hyperparameters and runs
        scikit-learn cross-validation.

        Parameters
        ----------
        environment : dict or None
            Bindings for variables contained in the DataOp plan. If not
            provided, the ``value``s passed when initializing ``var()`` are
            used.

        keep_subsampling : bool, default=False
            If True, and if subsampling has been configured (see
            :meth:`DataOp.skb.subsample`), use a subsample of the data. By
            default subsampling is not applied and all the data is used.

        kwargs : dict
            All other named arguments are forwarded to
            ``sklearn.model_selection.cross_validate``, except that
            scikit-learn's ``return_estimator`` parameter is named
            ``return_learner`` here.

        Returns
        -------
        dict
            Cross-validation results.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> import skrub

        >>> X_a, y_a = make_classification(random_state=0)
        >>> X, y = skrub.X(X_a), skrub.y(y_a)
        >>> pred = X.skb.apply(LogisticRegression(), y=y)
        >>> pred.skb.cross_validate(cv=2)['test_score']
        0    0.84
        1    0.78
        Name: test_score, dtype: float64

        Passing some data:

        >>> data = {'X': X_a, 'y': y_a}
        >>> pred.skb.cross_validate(data)['test_score']
        0    0.75
        1    0.90
        2    0.85
        3    0.65
        4    0.90
        Name: test_score, dtype: float64
        """
        if environment is None:
            environment = self.get_data()

        return cross_validate(
            self.make_learner(),
            environment,
            keep_subsampling=keep_subsampling,
            **kwargs,
        )

    @check_data_op
    def mark_as_X(self):
        """Mark this DataOp as being the ``X`` table.

        This is used for cross-validation and hyperparameter selection: operations
        done before :meth:`.skb.mark_as_X()` and :meth:`.skb.mark_as_y()` are executed
        on the entire data and cannot benefit from hyperparameter tuning.
        Returns a copy; the original DataOp is left unchanged.

        Returns
        -------
        The input DataOp, which has been marked as being ``X``

        See also
        --------
        :func:`skrub.X`
            ``skrub.X(value)`` can be used as a shorthand for
            ``skrub.var('X', value).skb.mark_as_X()``.

        Notes
        -----
        During cross-validation, all the previous steps are first executed,
        until X and y have been materialized. Then, those are split into
        training and testing sets. The following steps in the DataOp are
        fitted on the train data, and applied to test data, within each split.

        This means that any step that comes before ``mark_as_X()`` or
        ``mark_as_y()``, meaning that it is needed to compute X and y, sees the
        full dataset and cannot benefit from hyperparameter tuning. So we
        should be careful to start our learner by building X and y, and to use
        ``mark_as_X()`` and ``mark_as_y()`` as soon as possible.

        ``skrub.X(value)`` can be used as a shorthand for
        ``skrub.var('X', value).skb.mark_as_X()``.

        Note: this marks the DataOp in-place and also returns it.

        Examples
        --------
        >>> import skrub
        >>> orders = skrub.var('orders', skrub.datasets.toy_orders(split='all').orders)
        >>> features = orders.drop(columns='delayed', errors='ignore')
        >>> features.skb.is_X
        False
        >>> X = features.skb.mark_as_X()
        >>> X.skb.is_X
        True

        Note the original is left unchanged

        >>> features.skb.is_X
        False

        >>> y = orders['delayed'].skb.mark_as_y()
        >>> y.skb.is_y
        True

        Now if we run cross-validation:

        >>> from sklearn.dummy import DummyClassifier
        >>> pred = X.skb.apply(DummyClassifier(), y=y)
        >>> pred.skb.cross_validate(cv=2)['test_score']
        0    0.666667
        1    0.666667
        Name: test_score, dtype: float64

        First (outside of the cross-validation loop) ``X`` and ``y`` are
        computed. Then, they are split into training and test sets. Then the
        rest of the learner (in this case the last step, the
        ``DummyClassifier``) is evaluated on those splits.

        Please see the examples gallery for more information.
        """
        new = self._data_op._skrub_impl.__copy__()
        new.is_X = True
        return DataOp(new)

    @property
    def is_X(self):
        """Whether this DataOp has been marked with :meth:`.skb.mark_as_X()`."""
        return self._data_op._skrub_impl.is_X

    @check_data_op
    def mark_as_y(self):
        """Mark this DataOp as being the ``y`` table.

        This is used for cross-validation and hyperparameter selection: operations
        done before :meth:`.skb.mark_as_X()` and :meth:`.skb.mark_as_y()` are executed
        on the entire data and cannot benefit from hyperparameter tuning.
        Returns a copy; the original DataOp is left unchanged.

        Returns
        -------
        The input DataOp, which has been marked as being ``y``

        Notes
        -----
        During cross-validation, all the previous steps are first executed,
        until X and y have been materialized. Then, those are split into
        training and testing sets. The following steps in the DataOp plan are
        fitted on the train data, and applied to test data, within each split.

        This means that any step that comes before ``mark_as_X()`` or
        ``mark_as_y()``, meaning that it is needed to compute X and y, sees the
        full dataset and cannot benefit from hyperparameter tuning. So we
        should be careful to start our learner by building X and y, and to use
        ``mark_as_X()`` and ``mark_as_y()`` as soon as possible.

        Note: this marks the DataOp in-place and also returns it.

        See also
        --------
        :func:`skrub.y`
            ``skrub.y(value)`` can be used as a shorthand for
            ``skrub.var('y', value).skb.mark_as_y()``.

        Examples
        --------
        >>> import skrub
        >>> orders = skrub.var('orders', skrub.datasets.toy_orders(split='all').orders)
        >>> X = orders.drop(columns='delayed', errors='ignore').skb.mark_as_X()
        >>> delayed = orders['delayed']
        >>> delayed.skb.is_y
        False
        >>> y = delayed.skb.mark_as_y()
        >>> y.skb.is_y
        True

        Note the original is left unchanged

        >>> delayed.skb.is_y
        False

        Now if we run cross-validation:

        >>> from sklearn.dummy import DummyClassifier
        >>> pred = X.skb.apply(DummyClassifier(), y=y)
        >>> pred.skb.cross_validate(cv=2)['test_score']
        0    0.666667
        1    0.666667
        Name: test_score, dtype: float64

        First (outside of the cross-validation loop) ``X`` and ``y`` are
        computed. Then, they are split into training and test sets. Then the
        rest of the learner (in this case the last step, the
        ``DummyClassifier``) is evaluated on those splits.

        Please see the examples gallery for more information.
        """
        new = self._data_op._skrub_impl.__copy__()
        new.is_y = True
        return DataOp(new)

    @property
    def is_y(self):
        """Whether this DataOp has been marked with :meth:`.skb.mark_as_y()`."""
        return self._data_op._skrub_impl.is_y

    @check_data_op
    def set_name(self, name):
        """Give a name to this DataOp.

        Returns a modified copy.

        The name is displayed in the graph and reports so this can be useful to
        mark relevant parts of the learner.

        Moreover, the evaluation of this step can be bypassed and the result
        provided directly by providing a value for this name to ``eval()``,
        ``transform()``, ``predict()`` etc. (see examples)

        Parameters
        ----------
        name : str
            The name for this step. Must be unique within a learner. Cannot
            start with ``"_skrub_"``.

        Returns
        -------
        A new DataOp with the given name.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var('a', 1)
        >>> b = skrub.var('b', 2)
        >>> c = (a + b).skb.set_name('c')
        >>> c
        <c | BinOp: add>
        Result:
        ―――――――
        3
        >>> c.skb.name
        'c'
        >>> d = c * 10
        >>> d
        <BinOp: mul>
        Result:
        ―――――――
        30
        >>> d.skb.eval()
        30
        >>> d.skb.eval({'a': 10, 'b': 5})
        150

        We can override the result of ``c``. When we do, the operands ``a`` and
        ``b`` are not evaluated: evaluating ``c`` just returns the value we
        passed.

        >>> d.skb.eval({'c': -1}) # -1 * 10
        -10

        For DataOps that are not variables, the name can be set back to the
        default ``None``:

        >>> e = c.skb.set_name(None)
        >>> e.skb.name
        >>> c.skb.name
        'c'
        """
        check_name(name, isinstance(self._data_op._skrub_impl, Var))
        new = self._data_op._skrub_impl.__copy__()
        new.name = name
        return DataOp(new)

    @property
    def name(self):
        """A user-chosen name for the DataOp.

        The name is used for display, to retrieve a specific node inside the
        DataOp or to override its value. See :func:`DataOp.skb.set_name` for
        more information.
        """
        return self._data_op._skrub_impl.name

    def set_description(self, description):
        """Give a description to this DataOp.

        Returns a modified copy.

        The description can help document our learner. It is displayed in the
        execution report and can be retrieved from the ``.skb.description``
        attribute.

        Parameters
        ----------
        description : str
            The description

        Returns
        -------
        A new DataOp with the provided description.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var('a', 1)
        >>> b = skrub.var('b', 2)
        >>> c = (a + b).skb.set_description('the addition of a and b')
        >>> c
        <BinOp: add>
        Result:
        ―――――――
        3
        >>> c.skb.description
        'the addition of a and b'
        """
        new = self._data_op._skrub_impl.__copy__()
        new.description = description
        return DataOp(new)

    @property
    def description(self):
        """A user-defined description or comment about the DataOp.

        This can be set with ``.skb.set_description()`` and is displayed in the
        execution report.
        """
        return self._data_op._skrub_impl.description

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @property
    @check_data_op
    def applied_estimator(self):
        """Retrieve the estimator applied in the previous step, as a DataOp.

        Notes
        -----
        This attribute only exists for DataOp created with
        ``.skb.apply()``.

        Examples
        --------
        >>> import skrub
        >>> orders_df = skrub.datasets.toy_orders().X
        >>> features = skrub.X(orders_df).skb.apply(skrub.TableVectorizer())
        >>> fitted_vectorizer = features.skb.applied_estimator
        >>> fitted_vectorizer
        <AppliedEstimator>
        Result:
        ―――――――
        ApplyToFrame(transformer=TableVectorizer())

        Note that in order to restrict transformers to a subset of columns,
        they will be wrapped in a meta-estimator ``ApplyToFrame`` or
        ``ApplyToCols`` depending if the transformer is applied to each column
        separately or not. The actual transformer can be retrieved through the
        ``transformer_`` attribute of ``ApplyToFrame`` or ``transformers_``
        attribute of ``ApplyToCols`` (a dictionary mapping column names to the
        corresponding transformer).

        >>> fitted_vectorizer.transformer_
        <GetAttr 'transformer_'>
        Result:
        ―――――――
        TableVectorizer()

        >>> fitted_vectorizer.transformer_.column_to_kind_
        <GetAttr 'column_to_kind_'>
        Result:
        ―――――――
        {'ID': 'numeric', 'quantity': 'numeric', 'date': 'datetime', 'product': 'low_cardinality'}

        Here is an example of an estimator applied column-wise:

        >>> orders_df['description'] = [f'describe {p}' for p in orders_df['product']]
        >>> from skrub import selectors as s
        >>> out = skrub.X(orders_df).skb.apply(
        ...     skrub.StringEncoder(n_components=2), cols=s.string() - "date"
        ... )
        >>> fitted_vectorizer = out.skb.applied_estimator
        >>> fitted_vectorizer
        <AppliedEstimator>
        Result:
        ―――――――
        ApplyToCols(cols=(string() - cols('date')),
                     transformer=StringEncoder(n_components=2))
        >>> fitted_vectorizer.transformers_
        <GetAttr 'transformers_'>
        Result:
        ―――――――
        {'product': StringEncoder(n_components=2), 'description': StringEncoder(n_components=2)}
        """  # noqa: E501
        if not isinstance(self._data_op._skrub_impl, Apply):
            attribute_error(
                self,
                "applied_estimator",
                (
                    "`.skb.applied_estimator` only exists "
                    "on data_ops created with ``.skb.apply()``"
                ),
            )
        return DataOp(AppliedEstimator(self._data_op))
