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

We first generate some synthetic data with outliers to show how different scalers
transform the data, then we show how the choice of the scaler affects the prediction
performance of a simple neural network.

.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`
.. |StandardScaler| replace:: :class:`~sklearn.preprocessing.StandardScaler`
.. |QuantileTransformer| replace:: :class:`~sklearn.preprocessing.QuantileTransformer`

"""

# %%
# Plotting the effect of different scalers
# ----------------------------------------
#
# First, let's import the |SquashingScaler|, as well as the usual scikit-learn
# |StandardScaler| and |RobustScaler|.

# %%
import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

from skrub import SquashingScaler

np.random.seed(0)  # for reproducibility

# %%
# We then generate some random values sampling from a uniform distribution in the
# range ``[0, 1]``: note that this will produce values that are always positive.
# We then add some outliers in random positions in the array.
# Subtracting 50 allows to have some negative outliers in the data.

values = np.random.rand(100, 1)
n_outliers = 15
outlier_indices = np.random.choice(values.shape[0], size=n_outliers, replace=False)
values[outlier_indices] = np.random.rand(n_outliers, 1) * 100 - 50

# %%
# We then create one of each scaler and use them to scale the data independently.

# %%
squash_scaler = SquashingScaler()
squash_scaled = squash_scaler.fit_transform(values)

robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(values)

standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(values)

quantile_transformer = QuantileTransformer(n_quantiles=100)
quantile_scaled = quantile_transformer.fit_transform(values)


# %%
# To better show the effect of scaling, we create two plots, where we display the
# data points after sorting them in ascending order: in this way, all outliers
# are close to each other and with the proper sign.
# We create two subplots because the scale of the outliers is much larger than that
# of the inliers, which means that any detail in the inlier would be hidden.

# %%
import matplotlib.pyplot as plt

x = np.arange(values.shape[0])

fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))

ax = axs[0]
ax.plot(x, sorted(values), label="Original Values", linewidth=2.5)
ax.plot(x, sorted(squash_scaled), label="SquashingScaler")
ax.plot(x, sorted(robust_scaled), label="RobustScaler", linestyle="--")
ax.plot(x, sorted(standard_scaled), label="StandardScaler")
ax.plot(x, sorted(quantile_scaled), label="QuantileTransformer")

# Add a horizontal band in [-4, +4]
ax.axhspan(-4, 4, color="gray", alpha=0.15)
ax.set(title="Original data", xlim=[0, values.shape[0]], xlabel="Percentile")
ax.legend()

ax = axs[1]
ax.plot(x, sorted(values), label="Original Values", linewidth=2.5)
ax.plot(x, sorted(squash_scaled), label="SquashingScaler")
ax.plot(x, sorted(robust_scaled), label="RobustScaler", linestyle="--")
ax.plot(x, sorted(standard_scaled), label="StandardScaler")
ax.plot(x, sorted(quantile_scaled), label="QuantileTransformer")

ax.set(ylim=[-4, 4])
ax.set(title="In range [-4, 4]", xlim=[0, values.shape[0]], xlabel="Percentile")

# Highlight the bounds of the SquashingScaler
ax.axhline(y=3, alpha=0.2)
ax.axhline(y=-3, alpha=0.2)

fig.suptitle(
    "Comparison of different scalers on sorted data with outliers", fontsize=20
)
fig.supylabel("Value")

# %%
# The figure on the left immediately shows how the scale of the data may be completely
# off because of a minority of outliers, with the RobustScaler following the behavior
# of the original by retaining the larger scale of the outliers. On the other hand,
# both the SquashingScaler and the StandardScaler remain roughly in the ``[-4, 4]``
# range (highlighted in grey in the left figure).
#
# In the right figure we can then spot how the presence of outliers has completely
# flattened the curve produced by the StandardScaler, forcing the inliers to be
# very close to 0. The RobustScaler and the SquashingScaler instead follow the original
# data much more closely, after centering it on 0.
#
# Finally, the SquashingScaler performs a smooth clipping of outliers, constraining
# all values to be in the range ``[-max_absolute_value, max_absolute_value]``,
# where ``max_absolute_value`` is a parameter specified by the user (3 by default).

# %%
# Comparing numerical pre-processing methods on a neural network
# --------------------------------------------------------------
#
# In the second part of the example, we want to fit a neural network to predict
# employee salaries.
# The dataset contains numerical features, categorical features, text features,
# and dates.
# These features are first converted to numerical features using
# :class:`~skrub.TableVectorizer`. Since the encoded features are not normalized,
# we apply a numerical transformation to them.
#
# Finally, we fit a simple neural network and compare the R2 scores obtained with
# different numerical transformations.
#
# While we use a simple :class:`~sklearn.neural_network.MLPRegressor` here for
# simplicity, we generally recommend using better neural network implementations
# or tree-based models whenever low test errors are desired.

# %%
# We test the :class:`~skrub.SquashingScaler` against the
# :class:`~sklearn.preprocessing.StandardScaler` and the
# :class:`~sklearn.preprocessing.QuantileTransformer` from scikit-learn. We put
# each of these together in a pipeline with a TableVectorizer and a simple MLPRegressor.
# In the end, we print the R2 scores of each fold's validation set in a three-fold
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
# On the employee salaries dataset, the |SquashingScaler| performs
# better than |StandardScaler| and |QuantileTransformer| on all
# cross-validation folds.

# %%
