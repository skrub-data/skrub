class RejectColumn(Exception):
    """Used by single-column transformers to indicate they do not apply to a column.

    >>> import pandas as pd
    >>> from skrub._to_datetime import ToDatetime
    >>> df = pd.DataFrame(dict(a=['2020-02-02'], b=[12.5]))
    >>> ToDatetime().fit_transform(df['a'])
    0   2020-02-02
    Name: a, dtype: datetime64[ns]
    >>> ToDatetime().fit_transform(df['b'])
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Column 'b' does not contain strings.
    """

    pass
