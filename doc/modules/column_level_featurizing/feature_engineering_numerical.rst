.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |ToFloat| replace:: :class:`~skrub.ToFloat`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`
.. |RejectColumn| replace:: :class:`~skrub.core.RejectColumn`

.. _user_guide_feature_engineering_numeric_to_float:

Parsing and scaling numeric features
==========================================================

Converting heterogeneous numeric values to uniform float32
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many tabular datasets stored as csv files contain numeric information stored as
strings, mixed representations, locale-specific formats, or other non-standard
encodings.
Common issues include:

- Thousands separators (``1,234.56`` or ``1 234,56``)
- Use of apostrophes as separators (``4'567.89``)
- Negative numbers encoded inside parentheses (``(1,234.56)``)
- String columns that contain mostly numeric values, but with occasional invalid entries

To provide consistent numeric behavior, skrub includes the |ToFloat| transformer,
which standardizes all numeric-like columns to ``float32`` and handles a wide
range of real-world formatting issues automatically. Columns that cannot be parsed
are rejected with a |RejectColumn| exception.

Converting numbers to ``float32`` has the advantage of reducing memory pressure,
while retaining most of the information for training models.

>>> import pandas as pd
>>> from skrub import ToFloat
>>> s = pd.Series(['1.1', None, '3.3'], name='x')
>>> ToFloat().fit_transform(s)
0    1.1
1    NaN
2    3.3
Name: x, dtype: float32

If the transformer is fitted correctly, invalid values encountered at transform
time are replaced by ``NaN``:

>>> to_float.transform(pd.Series(['3.3', 'invalid'], name='x'))
0    3.3
1    NaN
Name: x, dtype: float32

Locale-dependent decimal separators can be handled by specifying the
``decimal`` and ``thousand`` parameter. Here we use comma as decimal separator, and
a space as thousands separators:

>>> s = pd.Series(["4 567,89", "12 567,89"], name="x")
>>> ToFloat(decimal=",", thousand=" ").fit_transform(s)
0    4567.8...
1    12567.8...
Name: x, dtype: float32

In some contexts, negative numbers may be represented with parentheses, instead of
using ``-``. This case is handled by the ``parentheses`` boolean parameter:

>>> s = pd.Series(["-1,234.56", "(1,234.56)"], name="neg")
>>> ToFloat(thousand=",", parentheses=True).fit_transform(s)
0   -1234.5...
1   -1234.5...
Name: neg, dtype: float32


.. _user_guide_squashing_scaler:

Robust scaling of numeric features using |SquashingScaler|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The |SquashingScaler| is a robust scaler for numeric features, particularly
useful when features include outliers (such as infinite values); missing values
are left unchanged (they are not interpolated).
The |SquashingScaler| centers and scales the data in such a way that outliers are
less likely to skew the final result compared to alternative methods.

Based on the specified ``quantile_range`` parameter, the scaler employs a scikit-learn
|RobustScaler| to rescale the values in a way that the quantile range occupies
interval of length two, centering the median to zero. It therefore ensures that
inliers are spread to a reasonable range. Afterwards, it uses a smooth clipping
function to ensure all values (including outliers and infinite values) are in the
range ``[-max_absolute_value, max_absolute_value]``. By default,
``max_absolute_value=3``.

>>> import pandas as pd
>>> import numpy as np
>>> from skrub import SquashingScaler

>>> X = pd.DataFrame(dict(col=[np.inf, -np.inf, 3, -1, np.nan, 2]))
>>> SquashingScaler(max_absolute_value=3).fit_transform(X)
array([[ 3.        ],
        [-3.        ],
        [ 0.49319696],
        [-1.34164079],
        [        nan],
        [ 0.        ]])

More information about the theory behind the scaler is available in the
|SquashingScaler| documentation, while this
:ref:`working example <sphx_glr_auto_examples_0100_squashing_scaler.py>` compares
different scalers when used on data that include outliers.
