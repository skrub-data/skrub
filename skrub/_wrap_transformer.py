from ._apply_to_cols import ApplyToCols
from ._apply_to_frame import ApplyToFrame
from .selectors import make_selector

__all__ = ["wrap_transformer"]


def wrap_transformer(
    transformer,
    selector,
    allow_reject=False,
    keep_original=False,
    rename_columns="{}",
    n_jobs=None,
    columnwise="auto",
):
    """Create a ``ApplyToCols`` or a ``ApplyToFrame`` transformer.

    The ``transformer`` is wrapped in a transformer that will apply it to part
    of the input dataframe.

    By default, if ``transformer`` is a single-column transformer
    (has a ``__single_column_transformer__`` attribute), it is wrapped in a
    ``ApplyToCols`` instance. Otherwise it is wrapped in a ``ApplyToFrame``
    instance.

    This default choice can be overridden by passing ``columnwise=True`` to
    force the use of ``ApplyToCols`` or ``columnwise=False`` to force the use
    of ``ApplyToFrame``.

    Parameters
    ----------
    transformer : transformer (single-column or not)
        The transformer to wrap.

    selector : skrub selector
        The columns to which the transformer will be applied.

    allow_reject : bool, default=False
        Whether to allow column rejections. Only used when the result is an
        instance of ``ApplyToCols``, see this class' docstring for details.

    keep_original : bool, default=False
        Whether to retain the original columns in transformed output. See the
        documentation of ``ApplyToCols`` or ``ApplyToFrame`` for details.

    rename_columns : str, default='{}'
        Format string applied to output column names. See the documentation of
        ``ApplyToCols`` or ``ApplyToFrame`` for details.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Only used when the result is an
        instance of ``ApplyToCols``, see this class' docstring for details.

    columnwise : 'auto' or bool, default='auto'
        Whether to create a ``ApplyToCols`` or ``ApplyToFrame`` instance. By
        default, ``ApplyToCols`` is used if ``transformer`` has a
        ``__single_column_transformer__`` attribute and ``ApplyToFrame``
        otherwise. Pass ``columnwise=True`` to force using ``ApplyToCols`` and
        ``columnwise=False`` to force using ``ApplyToFrame``. Note that forcing
        ``columnwise=False`` for a single-column transformer will most likely
        cause an error during ``fit``, and forcing ``columnwise=True`` for a
        regular transformer is only appropriate if the transformer can be
        fitted on a dataframe with only one column (this is the case for most
        preprocessors such as ``OrdinalEncoder`` or ``StandardScaler``).

    Returns
    -------
    Wrapped transformer
        A ``ApplyToCols`` or ``ApplyToFrame`` instance initialized with the
        input ``transformer``.

    Examples
    --------
    >>> from skrub._wrap_transformer import wrap_transformer
    >>> from skrub._to_datetime import ToDatetime
    >>> from skrub import selectors as s
    >>> from sklearn.preprocessing import OrdinalEncoder

    >>> wrap_transformer(ToDatetime(), s.all())
    ApplyToCols(transformer=ToDatetime())
    >>> wrap_transformer(OrdinalEncoder(), s.string())
    ApplyToFrame(cols=string(), transformer=OrdinalEncoder())
    >>> wrap_transformer(OrdinalEncoder(), s.string(), columnwise=True, n_jobs=4)
    ApplyToCols(cols=string(), n_jobs=4, transformer=OrdinalEncoder())
    """
    selector = make_selector(selector)

    if isinstance(columnwise, str) and columnwise == "auto":
        columnwise = hasattr(transformer, "__single_column_transformer__")

    if columnwise:
        return ApplyToCols(
            transformer,
            cols=selector,
            allow_reject=allow_reject,
            keep_original=keep_original,
            rename_columns=rename_columns,
            n_jobs=n_jobs,
        )
    return ApplyToFrame(
        transformer,
        cols=selector,
        keep_original=keep_original,
        rename_columns=rename_columns,
    )
