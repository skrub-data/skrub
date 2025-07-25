"""
SquashingScaler: Robust numerical preprocessing for neural networks
===================================================================

The following example illustrates the use of the :class:`~skrub.SquashingScaler`, a
transformer that can rescale and squash numerical features to a range that works well
with neural networks and perhaps also other related models. Its basic idea is to
rescale the features based on quantile statistics (to be robust to outliers), and then
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

While we use a simple :class:`~sklearn.neural_network.MLPRegressor` here for simplicity,
we generally recommend using better neural network implementations or tree-based models
whenever low test errors are desired.
"""

# %%
# Comparing numerical preprocessings
# ----------------------------------
# We test the :class:`~skrub.SquashingScaler` against the
# :class:`~sklearn.preprocessing.StandardScaler` and the
# :class:`~sklearn.preprocessing.QuantileTransformer` from scikit-learn. We put each of
# these together in a pipeline with a TableVectorizer and a simple MLPRegressor. In the
# end, we print the R2 scores of each fold's validation set in a three-fold
# cross-validation.
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from skrub import DatetimeEncoder, SquashingScaler, TableVectorizer
from skrub.datasets import fetch_employee_salaries

np.random.seed(0)
data = fetch_employee_salaries()

for num_transformer in [
    StandardScaler(),
    QuantileTransformer(output_distribution="normal", random_state=0),
    SquashingScaler(),
]:
    pipeline = make_pipeline(
        TableVectorizer(datetime=DatetimeEncoder(periodic_encoding="circular")),
        num_transformer,
        TransformedTargetRegressor(
            # We use lbfgs for faster convergence
            MLPRegressor(solver="lbfgs", max_iter=100),
            transformer=StandardScaler(),
        ),
    )
    with warnings.catch_warnings():
        # Ignore warnings about the MLPRegressor not converging
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        scores = cross_validate(pipeline, data.X, data.y, cv=3, scoring="r2")

    print(
        f"Cross-validation R2 scores for {num_transformer.__class__.__name__}"
        f" (higher is better):\n{scores['test_score']}\n"
    )

# %%
# On the employee salaries dataset, the SquashingScaler performs
# better than StandardScaler and QuantileTransformer on all
# cross-validation folds.
