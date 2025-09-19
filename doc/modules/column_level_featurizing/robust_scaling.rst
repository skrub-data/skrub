.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`

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
:ref:`working example <sphx_glr_auto_examples_10_squashing_scaler.py>` compares
different scalers when used on data that include outliers.
