"""Base class for single-column transformers."""

import functools
import re
import textwrap

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd

__all__ = ["SingleColumnTransformer", "RejectColumn"]

_SINGLE_COL_LINE = (
    "``{class_name}`` is a type of single-column transformer. Unlike most scikit-learn"
    " estimators, its ``fit``, ``transform`` and ``fit_transform`` methods expect a"
    " single column (a pandas or polars Series) rather than a full dataframe. To apply"
    " this transformer to one or more columns in a dataframe, use it as a parameter in"
    " a ``skrub.ApplyToCols`` or a ``skrub.TableVectorizer``."
    " To apply to all columns::\n\n"
    "   ApplyToCol({class_name}())\n\n"
    "To apply to selected columns::\n\n"
    "   ApplyToCols({class_name}(), cols=['col_name_1', 'col_name_2'])"
)
_SINGLE_COL_PARAGRAPH = textwrap.fill(
    _SINGLE_COL_LINE, initial_indent="    ", subsequent_indent="    "
)
_SINGLE_COL_NOTE = f".. note::\n\n{_SINGLE_COL_PARAGRAPH}\n"


class RejectColumn(ValueError):
    """Used by single-column transformers to indicate they do not apply to a column.

    >>> import pandas as pd
    >>> from skrub import ToDatetime
    >>> df = pd.DataFrame(dict(a=['2020-02-02'], b=[12.5]))
    >>> ToDatetime().fit_transform(df['a'])
    0   2020-02-02
    Name: a, dtype: datetime64[...]
    >>> ToDatetime().fit_transform(df['b'])
    Traceback (most recent call last):
        ...
    skrub._single_column_transformer.RejectColumn: Column 'b' does not contain strings.
    """

    pass


class SingleColumnTransformer(BaseEstimator):
    """Base class for single-column transformers.

    Such transformers are applied independently to each column by
    ``ApplyToCols``; see the docstring of ``ApplyToCols`` for more
    information.

    Single-column transformers are not required to inherit from this class in
    order to work with ``ApplyToCols``, however doing so avoids some
    boilerplate:

        - The required ``__single_column_transformer__`` attribute is set.
        - ``fit`` is defined (calls ``fit_transform`` and discards the result).
        - ``fit``, ``transform`` and ``fit_transform`` are wrapped to check
          that the input is a single column and raise a ``ValueError`` with a
          helpful message when it is not.
        - A note about single-column transformers (vs dataframe transformers)
          is added after the summary line of the docstring.

    Subclasses must define ``fit_transform`` and ``transform`` (or inherit them
    from another superclass).
    """

    __single_column_transformer__ = True

    def fit(self, column, y=None, **kwargs):
        """Fit the transformer.

        This default implementation simply calls ``fit_transform()`` and
        returns ``self``.

        Subclasses should implement ``fit_transform`` and ``transform``.

        Parameters
        ----------
        column : a pandas or polars Series
            Unlike most scikit-learn transformers, single-column transformers
            transform a single column, not a whole dataframe.

        y : column or dataframe
            Prediction targets.

        **kwargs
            Extra named arguments are passed to ``self.fit_transform()``.

        Returns
        -------
        self
            The fitted transformer.
        """
        self.fit_transform(column, y=y, **kwargs)
        return self

    def _check_single_column(self, column, function_name):
        class_name = self.__class__.__name__
        if sbd.is_dataframe(column):
            raise ValueError(
                f"``{class_name}.{function_name}`` should be passed a single column,"
                " not a dataframe. " + _SINGLE_COL_LINE.format(class_name=class_name)
            )
        if not sbd.is_column(column):
            raise ValueError(
                f"``{class_name}.{function_name}`` expects the first argument X "
                "to be a column (a pandas or polars Series). "
                f"Got X with type: {column.__class__.__name__}."
            )
        return column

    def __init_subclass__(subclass, **kwargs):
        super().__init_subclass__(**kwargs)
        if subclass.__doc__ is not None:
            subclass.__doc__ = _insert_after_first_paragraph(
                subclass.__doc__,
                _SINGLE_COL_NOTE.format(class_name=subclass.__name__),
            )
        for method in "fit", "fit_transform", "transform", "partial_fit":
            if method in subclass.__dict__:
                wrapped = _wrap_add_check_single_column(getattr(subclass, method))
                setattr(subclass, method, wrapped)

    def get_feature_names_out(self, input_features=None):
        """Get the output feature names.

        Parameters
        -----------
        input_features : array-like of str, default=None
            Input feature names. Ignored.

        Returns
        --------
        all_outputs_
            The names of the output features.
        """
        check_is_fitted(self, "all_outputs_")
        return self.all_outputs_


def _wrap_add_check_single_column(f):
    # as we have only a few predefined functions to handle, using their exact
    # name and signature in the wrapper definition gives better tracebacks and
    # autocompletion than just functools.wraps / setting __name__ and
    # __signature__
    if f.__name__ == "fit":

        @functools.wraps(f)
        def fit(self, X, y=None, **kwargs):
            self._check_single_column(X, f.__name__)
            return f(self, X, y=y, **kwargs)

        return fit
    elif f.__name__ == "partial_fit":

        @functools.wraps(f)
        def partial_fit(self, X, y=None, **kwargs):
            self._check_single_column(X, f.__name__)
            return f(self, X, y=y, **kwargs)

        return partial_fit

    elif f.__name__ == "fit_transform":

        @functools.wraps(f)
        def fit_transform(self, X, y=None, **kwargs):
            self._check_single_column(X, f.__name__)
            return f(self, X, y=y, **kwargs)

        return fit_transform
    else:
        assert f.__name__ == "transform", f.__name__

        @functools.wraps(f)
        def transform(self, X, **kwargs):
            self._check_single_column(X, f.__name__)
            return f(self, X, **kwargs)

        return transform


def _insert_after_first_paragraph(document, text_to_insert):
    split_doc = document.splitlines(True)
    indent = min(
        (
            len(m.group(1))
            for line in split_doc[1:]
            if (m := re.match(r"^( *)\S", line)) is not None
        ),
        default=0,
    )
    doc_lines = iter(split_doc)
    output_lines = []
    for line in doc_lines:
        output_lines.append(line)
        if line.strip():
            break
    for line in doc_lines:
        output_lines.append(line)
        if not line.strip():
            break
    else:
        output_lines.append("\n")
    for line in text_to_insert.splitlines(True):
        output_lines.append(line if not line.strip() else " " * indent + line)
    output_lines.append("\n")
    output_lines.extend(doc_lines)
    return "".join(output_lines)
