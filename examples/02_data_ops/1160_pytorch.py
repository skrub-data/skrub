"""
Using PyTorch (via skorch) in DataOps
======================================

This example shows how to wrap a PyTorch model with skorch and plug it into a
skrub DataOps plan.

.. note::
    This example requires the optional dependencies ``torch`` and ``skorch``.

The main goal here is to show the *integration pattern*:

- **PyTorch** defines the model (an ``nn.Module``)
- **skorch** wraps it as a scikit-learn compatible estimator
- **skrub DataOps** builds a plan and can tune skorch (and therefore PyTorch)
  hyperparameters using the skrub choices.
"""

# %%
# Loading the data
# =================
#
# We use scikit-learn's digits dataset because it is small and ships with
# scikit-learn. Each sample is an 8x8 grayscale image of a
# handwritten digit, encoded as 64 pixel intensity values and displays a
# number from 0 to 9.
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(set(y))}")

# %%
# Start of the DataOps plan
# ==========================
#
# We start the DataOps plan by creating the skrub variables X and y.
import skrub

X = skrub.X(X)
y = skrub.y(y)

# %%
# Data preprocessing
# ==================
#
# We start by normalizing the pixel values to [0, 1] by first
# computing the global max value and then dividing the pixel values
# by this max value. Importantly, we freeze the max value (scaling factor)
# after fitting so that the same rescaling is applied later when we use our
# dataop for prediction on new (test) data.
#
# A convolutional network expects images with shape (N, C, H, W) where:
#
# - N: number of samples
# - C: number of color channels (1 for grayscale)
# - H, W: image height and width
#
# So we reshape the images to (N, 1, 8, 8) for the CNN. The -1 means the first
# dimension (N) is inferred automatically from the array size.
#
# The advantage of using DataOps is that the preprocessing steps are tracked
# in the plan and will be automatically applied during prediction.

max_value = X.max().skb.freeze_after_fit()
X_scaled = X / max_value
X_reshaped = X_scaled.reshape(-1, 1, 8, 8).astype("float32")
X_reshaped.skb.draw_graph()


# %%
# Building a NN Classifier
# =========================
#
# We'll build a tiny CNN using PyTorch and wrap it with skorch to make it
# scikit-learn compatible. The architecture uses a single convolution + pooling
# stage and a small MLP head. The architectural choices below are meant to be:
#
# - **standard**: 3x3 convolutions and 2x2 max-pooling are very common
# - **small**: the dataset and images are tiny, so we keep the model tiny too
#
# If you want more background on CNN building blocks and how convolution/pooling
# changes tensor shapes, see the CS231n notes:
# https://cs231n.github.io/convolutional-networks/
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TinyCNN(nn.Module):
    def __init__(self, conv_channels: int = 8, hidden_units: int = 32):
        super().__init__()
        self.conv_channels = conv_channels
        self.hidden_units = hidden_units

        # 2-level CNN with 2x2 max-pooling
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # input shape = (8,8) -> conv1: (8,8) -> conv2: (8,8) -> pool: (4,4)
        image_shape_after_conv = 4 * 4

        # MLP head
        self.fc1 = nn.Linear(conv_channels * image_shape_after_conv, hidden_units)
        self.dropout = nn.Dropout(p=0.25)  # Regularization to avoid overfitting
        self.fc2 = nn.Linear(hidden_units, 10)  # 10 digit classes (0..9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# %%
# Skorch provides scikit-learn compatible wrappers around torch training loops.
# That makes the torch model usable by skrub DataOps (and scikit-learn tools in
# general).
#
# We use :func:`skrub.choose_from()` to define hyperparameters that the DataOps
# grid search will tune: conv_channels, hidden_units, and max_epochs.
# The other parameters are set to common choices for this task and training data size.

from skorch import NeuralNetClassifier

device = "cpu"  # use "cuda" or "mps" if available

net = NeuralNetClassifier(
    module=TinyCNN,
    # These choices are intentionally small so the example runs quickly.
    module__conv_channels=skrub.choose_from([8, 16], name="conv_channels"),
    module__hidden_units=skrub.choose_from([8, 16, 32], name="hidden_units"),
    max_epochs=skrub.choose_from([10, 15], name="max_epochs"),
    optimizer__lr=0.01,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=device,
    train_split=None,  # We'll use skrub's grid search for validation
    verbose=0,
)


# %%
# Tuning the model's hyperparameters with DataOps
# ===============================================
#
# We integrate the model into the DataOps plan. First, we
# convert the target labels to integers for the loss computation
# and apply the model to the preprocessed X and y.

y_int = y.astype("int64")
predictor = X_reshaped.skb.apply(net, y=y_int)
predictor.skb.draw_graph()

# %%
# Finally, we use 4-fold cross-validation for the hyperparameter
# tuning on our DataOps plan.

from sklearn.model_selection import KFold

cv = KFold(n_splits=4, shuffle=True, random_state=42)
search = predictor.skb.make_grid_search(
    cv=cv,
    fitted=True,
    n_jobs=-1,
)
print("\nSearch results:")
print(search.results_.to_string(index=False))

# %%
# Let's take a better look at the well-performing models by looking
# at the parallel coordinates plot. We filter to models with
# score >= 0.94 to focus on the top-performing configurations.

fig = search.plot_results(min_score=0.94)
fig

# %%
# Interpreting the results
# ========================
#
# Looking at the search results, we can observe several patterns:
#
# - **Model capacity matters**: Larger configurations with ``conv_channels=16``
#   and ``hidden_units=32`` tend to perform best. Smaller models with
#   ``conv_channels=8`` and/or ``hidden_units=8`` perform significantly worse,
#   indicating that the task benefits from increased model capacity.
# - **More epochs generally help**: Configurations with ``max_epochs=15`` tend to
#   perform slightly better than those with ``max_epochs=10``, though the gains
#   are modest compared to architectural changes.

# %%
# Conclusion
# ==========
#
# In this example, we've shown how to use **PyTorch** and **skorch** within
# skrub DataOps. The key steps were:
#
# 1. Define a PyTorch ``nn.Module`` (our ``TinyCNN``)
# 2. Wrap it with skorch's ``NeuralNetClassifier`` to make it scikit-learn compatible
# 3. Use :func:`skrub.choose_from()` to specify hyperparameters for tuning
# 4. Integrate it into a DataOps plan and use grid search to find the best configuration
#
# This pattern lets you leverage PyTorch's flexibility for model definition while
# benefiting from skrub's hyperparameter tuning and data preprocessing capabilities.
#
# .. seealso::
#
#    * :ref:`example_tuning_pipelines`: Learn more about using
#      ``skrub.choose_from()`` and other choice objects to tune hyperparameters
#      in DataOps plans.
#    * :ref:`example_optuna_choices`: Discover how to use Optuna as a backend
#      for more sophisticated hyperparameter search strategies with skrub DataOps.
