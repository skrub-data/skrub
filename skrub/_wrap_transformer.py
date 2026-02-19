from ._apply_on_each_col import ApplyToEachCol
from ._apply_sub_frame import ApplyToSubFrame
from ._single_column_transformer import is_single_column_transformer
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
    """Create a ``ApplyToEachCol`` or a ``ApplyToSubFrame`` transformer.

    The ``transformer`` is wrapped in a transformer that will apply it to part
    of the input dataframe.

    By default, if ``transformer`` is a single-column transformer
    (has a ``__single_column_transformer__`` attribute), it is wrapped in a
    ``ApplyToEachCol`` instance. Otherwise it is wrapped in a ``ApplyToSubFrame``
    instance.

    This default choice can be overridden by passing ``columnwise=True`` to
    force the use of ``ApplyToEachCol`` or ``columnwise=False`` to force the use
    of ``ApplyToSubFrame``.

    Parameters
    ----------
    transformer : transformer (single-column or not)
        The transformer to wrap.

    selector : skrub selector
        The columns to which the transformer will be applied.

    allow_reject : bool, default=False
        Whether to allow column rejections. Only used when the result is an
        instance of ``ApplyToEachCol``, see this class' docstring for details.

    keep_original : bool, default=False
        Whether to retain the original columns in transformed output. See the
        documentation of ``ApplyToEachCol`` or ``ApplyToSubFrame`` for details.

    rename_columns : str, default='{}'
        Format string applied to output column names. See the documentation of
        ``ApplyToEachCol`` or ``ApplyToSubFrame`` for details.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Only used when the result is an
        instance of ``ApplyToEachCol``, see this class' docstring for details.

    columnwise : 'auto' or bool, default='auto'
        Whether to create a ``ApplyToEachCol`` or ``ApplyToSubFrame`` instance. By
        default, ``ApplyToEachCol`` is used if ``transformer`` has a
        ``__single_column_transformer__`` attribute and ``ApplyToSubFrame``
        otherwise. Pass ``columnwise=True`` to force using ``ApplyToEachCol`` and
        ``columnwise=False`` to force using ``ApplyToSubFrame``. Note that forcing
        ``columnwise=False`` for a single-column transformer will most likely
        cause an error during ``fit``, and forcing ``columnwise=True`` for a
        regular transformer is only appropriate if the transformer can be
        fitted on a dataframe with only one column (this is the case for most
        preprocessors such as ``OrdinalEncoder`` or ``StandardScaler``).

    Returns
    -------
    Wrapped transformer
        A ``ApplyToEachCol`` or ``ApplyToSubFrame`` instance initialized with the
        input ``transformer``.

    Examples
    --------
    >>> from skrub._wrap_transformer import wrap_transformer
    >>> from skrub._to_datetime import ToDatetime
    >>> from skrub import selectors as s
    >>> from sklearn.preprocessing import OrdinalEncoder

    >>> wrap_transformer(ToDatetime(), s.all())
    ApplyToEachCol(transformer=ToDatetime())
    >>> wrap_transformer(OrdinalEncoder(), s.string())
    ApplyToSubFrame(cols=string(), transformer=OrdinalEncoder())
    >>> wrap_transformer(OrdinalEncoder(), s.string(), columnwise=True, n_jobs=4)
    ApplyToEachCol(cols=string(), n_jobs=4, transformer=OrdinalEncoder())
    """
    selector = make_selector(selector)

    if isinstance(columnwise, str) and columnwise == "auto":
        columnwise = is_single_column_transformer(transformer)

    if columnwise:
        return ApplyToEachCol(
            transformer,
            cols=selector,
            allow_reject=allow_reject,
            keep_original=keep_original,
            rename_columns=rename_columns,
            n_jobs=n_jobs,
        )
    return ApplyToSubFrame(
        transformer,
        cols=selector,
        keep_original=keep_original,
        rename_columns=rename_columns,
    )
