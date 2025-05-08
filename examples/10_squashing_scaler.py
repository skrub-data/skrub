"""
SquashingScaler: Robust numerical preprocessing for neural networks
==============================================================

The following example illustrates the use of the :class:`~skrub.SquashingScaler`, a
transformer that can rescale and squash numerical features to a range that works well
with neural networks and perhaps also other related models. Its basic idea is to
rescale the features based on quantile statistics, to be robust to outliers, and then
perform a smooth squashing function to limit the outputs to a pre-defined range.
This transform has been found to even work well when applied to one-hot encoded
features.

In this example, we want to fit a neural network to predict employee salaries.
The dataset contains numerical features, categorical features, text features, and dates.
These features are first converted to numerical features using
:class:`~skrub.TableVectorizer`. Since the encoded features are not normalized, we apply
a numerical transformation to them.
Finally, we fit a simple neural network and compare the R2 scores obtained with
different numerical transformations.
While we use a simple MLPRegressor here for simplicity,
we generally recommend using better neural network implementations or tree-based models
whenever low test errors are desired.
"""

import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from skrub import DatetimeEncoder, SquashingScaler, TableVectorizer
from skrub.datasets import fetch_employee_salaries

######################################################################
# Comparing numerical preprocessings
# -----------------
# We test the :class:`~skrub.SquashingScaler` vs the `StandardScaler`
# and the `QuantileTransformer` from scikit-learn.
# We put each of these together in a pipeline with a TableVectorizer
# and a simple MLPRegressor.
# In the end, we print the R2 scores of each fold's validation set
# in a three-fold cross-validation.

for num_transformer in [
    StandardScaler(),
    QuantileTransformer(output_distribution="normal", random_state=0),
    SquashingScaler(),
]:
    np.random.seed(0)
    data = fetch_employee_salaries()

    model = TransformedTargetRegressor(MLPRegressor(), transformer=StandardScaler())

    pipeline = Pipeline(
        steps=[
            (
                "tv",
                TableVectorizer(datetime=DatetimeEncoder(periodic_encoding="circular")),
            ),
            ("num", num_transformer),
            ("model", model),
        ]
    )

    with warnings.catch_warnings():
        # ignore warning about unknown categories
        warnings.simplefilter("ignore", category=UserWarning)

        scores = cross_val_score(pipeline, data.X, data.y, cv=3, scoring="r2")

    print(
        f"Cross-validation R2 scores for {num_transformer.__class__.__name__}"
        f" (higher is better):\n{scores}"
    )

######################################################################
# Result
# -----------------
# On the employee salaries dataset, the SquashingScaler performs
# better than StandardScaler and QuantileTransformer on all
# cross-validation folds.
