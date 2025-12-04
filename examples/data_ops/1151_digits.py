"""
Handwritten digit classification with skrub
============================================

This example demonstrates how to use skrub's Data Ops to build a
machine learning pipeline for handwritten digit classification. We'll cover:

1. Loading and exploring the UCI ML hand-written digits dataset
2. Building a basic classification model with skrub Data Ops
3. Performing hyperparameter tuning with multiple algorithms
4. Comparing model performance before and after tuning

This example is inspired by the official scikit-learn example:
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
"""

# %%
# Loading the Data
# ================
#
# We use the UCI ML hand-written digits dataset provided by scikit-learn.
# This dataset contains 8x8 pixel images of handwritten digits (0-9) that have
# been flattened to 64-dimensional feature vectors. Our goal is to build a
# classifier that can accurately predict the digit from the pixel values.

from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(set(y))}")
# %%
# Visualizing Sample Digits
# ==========================
#
# Let's visualize some examples from the dataset to understand what we're
# working with. Each image is an 8x8 grid of pixel intensities representing
# a handwritten digit.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, image, label in zip(axes.flatten(), X, y):
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label: {label}")
plt.tight_layout()
# %%
# Building a Baseline Model
# ==========================
#
# Now we'll build our first model using skrub's Data Ops framework.
# First, we define our X and y variables as Data Ops.

import skrub

X = skrub.X(X)
y = skrub.y(y)

# %%
# Starting with a Simple SVM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We begin with a Support Vector Machine (SVM) classifier. The skrub
# Data Ops framework allows us to define our model and apply it to the data
# in a declarative way using the apply method.

from sklearn.svm import SVC

model = SVC()
predictions = X.skb.apply(model, y=y)

# %%
# After defining the computational plan with apply, we create a learner object
# and evaluate it using cross-validation. This gives us a baseline performance
# metric for our SVM classifier.

learner = predictions.skb.make_learner(fitted=True)

cv_results = skrub.cross_validate(learner, environment=predictions.skb.get_data())
print(f"Baseline SVM - Mean CV accuracy: {cv_results['test_score'].mean():.4f}")

# %%
# Hyperparameter Tuning with Multiple Models
# ============================================
#
# To improve performance, we'll now explore multiple algorithms and their
# hyperparameters using skrub's hyperparameter tuning capabilities.
# The choose_from and choose_* functions define a search space over which
# we can perform randomized search.

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# %%
# Define a search space with three different algorithms:
# 1. **SVM with C and gamma tuning**: Regularization and kernel parameters
# 2. **Random Forest**: Number of trees and tree depth
# 3. **Histogram Gradient Boosting**: Iterations, depth, and learning rate

model = skrub.choose_from(
    {
        "SVC": SVC(
            C=skrub.choose_float(0.1, 10.0, name="C", log=True),
            gamma=skrub.choose_float(0.001, 1.0, name="gamma", log=True),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=skrub.choose_int(10, 200, name="n_estimators"),
            max_depth=skrub.choose_int(5, 50, name="max_depth_rf"),
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=skrub.choose_int(50, 200, name="max_iter"),
            max_depth=skrub.choose_int(3, 20, name="max_depth_hgb"),
            learning_rate=skrub.choose_float(0.01, 0.5, log=True, name="learning_rate"),
        ),
    }
)

predictions = X.skb.apply(model, y=y)

# %%
# Running Randomized Search
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We perform a randomized search over the hyperparameter space. This involves:
# - Randomly sampling 50 different hyperparameter combinations
# - Using 5-fold cross-validation to evaluate each combination
# - Running evaluations in parallel (n_jobs=-1)

search = predictions.skb.make_randomized_search(
    n_iter=50, cv=5, random_state=42, n_jobs=-1, fitted=True
)

# %%
# Visualizing Search Results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's plot the search results to see how different hyperparameter
# combinations affected performance:

search.plot_results()

# %%
# Comparing Results
# ~~~~~~~~~~~~~~~~~
#
# Finally, we compare the performance of our baseline model with the
# best model found during hyperparameter tuning:

best_learner = search.best_learner_
cv_results_search = skrub.cross_validate(
    best_learner, environment=predictions.skb.get_data()
)

print("Mean CV accuracies comparison:")
print(f"Before hyperparameter tuning: {cv_results['test_score'].mean():.4f}")
print(f"After hyperparameter tuning:  {cv_results_search['test_score'].mean():.4f}")

# %%
# Related examples
# ====================
# You can find more information on how to tune hyperparameters with skrub in
# the hyperparameter tuning
# :ref:`example <sphx_glr_auto_examples_data_ops_1130_choices.py>`,
# and in the User Guide section on
# :ref:`Hyperparameter Tuning <user_guide_data_ops_hyperparameter_tuning>`.
