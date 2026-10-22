"""
.. _example_formula:

Formula-like feature engineering with skrub DataOps
====================================================

In R, and in the Python package patsy, the features of a statistical model are
described by a *formula*, such as::

    cardio ~ bs(bmi, df=4) + age + gender + ap_hi + ap_lo

where ``bs`` stands for B-splines, a flexible way to model non-linear
relationships between a predictor and the response.

:ref:`DataOps <user_guide_data_ops_index>` can do the same without a formula
string: the features are built with ordinary Python code, so we keep the full
power of Python and pandas for feature engineering, combined with
scikit-learn transformers -- which matters to separate train and test data
properly.

The benefit of this approach is that we can write sophisticated feature
engineering code in plain Python, but drawing in any scikit-learn compatible
transformer for machine learning components.

.. currentmodule:: skrub

"""

# %%
# The data
# --------
#
# We use the Cardiovascular Disease dataset from OpenML: 70,000 patients
# described by clinical measurements, and a binary target, ``cardio``. The
# OpenML version stores several columns as pandas categories holding integers;
# we turn them into plain numbers so that we can compute with them below.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer, StandardScaler

import skrub

df = fetch_openml(data_id=45547, as_frame=True).frame
df = df.astype({c: "int64" for c in df.select_dtypes("category").columns})

skrub.TableReport(df)

# %%
# Holding out a test set
# ----------------------
#
# Before writing any feature code, we keep 20% of the patients aside. Whatever
# we build next is fitted on the training rows only, and the test rows are
# used at the end to see how it behaves on patients it has never seen.

train, test = train_test_split(df, test_size=0.2, random_state=0)
train.shape, test.shape

# %%
# Building the features
# ---------------------
#
# We build the features with a skrub DataOps. We start from the training table,
# separate the input and the target, and tell skrub which is which:
# :meth:`.skb.mark_as_X() <DataOp.skb.mark_as_X>` and
# :meth:`.skb.mark_as_y() <DataOp.skb.mark_as_y>`. The target is only needed
# to fit the model; predictions only require the input columns.

data = skrub.var("data", train)

# errors="ignore": new patients come without their target column
X = data.drop(columns="cardio", errors="ignore").skb.mark_as_X()
y = data["cardio"].skb.mark_as_y()

# %%
# Now the features themselves. They are ordinary pandas expressions on the
# columns. The body mass index divides the weight by the squared height: a
# ratio a linear model would never find on its own from the two columns. The
# age, given in days, becomes a number of years.

X = X.assign(bmi=X["weight"] / (X["height"] / 100) ** 2)
X = X.assign(age=X["age"] / 365.25)

features = X[["age", "gender", "ap_hi", "ap_lo", "bmi"]]
features

# %%
# The effect of the BMI is unlikely to be a straight line, so the model does
# not receive the BMI itself but a B-spline basis (the ``bs`` above): a set of
# smooth bumps that a linear model can mix to bend the relationship.
#
# This step is *fitted*: the transformer chooses the position of its knots by
# looking at the BMI values it receives. That is where the early split pays
# off -- the knots are learned from the training patients, and the test
# patients are then projected onto that same basis. Had it seen the whole
# dataset beforehand, the test rows would have shaped the features used to
# judge them.

splines = features.skb.apply(SplineTransformer(n_knots=4), cols=["bmi"]).skb.set_name(
    "splines"
)
splines

# %%
# The predictor
# --------------
#
# The last step is the predictor. We use a linear model -- a logistic
# regression -- as a linear model is great for interpretability.
# The spline basis lets it capture a non-linear effect. A scaler first puts
# the columns on comparable scales.

pred = (
    splines.skb.apply(StandardScaler())
    .skb.set_name("scaler")
    .skb.apply(LogisticRegression(), y=y)
    .skb.set_name("model")
)

# %%
# Cross-validation: Evaluating the pipeline
# -----------------------------------------
#
# With the full pipeline assembled, we can evaluate it with cross-validation
# on the training rows, leaving the test set untouched. The splines have
# parameters learned from the data -- the position of their knots -- and
# DataOps make sure that, in each fold, they are learned on the training part
# only and that the held-out part is projected onto the same basis.

pred.skb.cross_validate(cv=5, scoring="roc_auc").round(3)

# %%
# A learner that can be applied to new data
# -------------------------------------------
#
# To get a learner that can be applied to new data, we call
# ``make_learner(fitted=True)``. This fits all the transformations at once:
# the BMI, the spline basis, the scaler, and the regression, on the data that
# was provided in the beginning, when we defined the 'train' variable.
learner = pred.skb.make_learner(fitted=True)

# %%
# The learner is an ordinary predictor: we give it a table with the same
# input columns as the training table, and it returns a prediction for every
# row. Here are the held-out patients, with only their measurements -- the
# target is not something we would have for new patients.

new_patients = test.drop(columns="cardio")
probabilities = learner.predict_proba({"data": new_patients})[:, 1]

print("predictions:", learner.predict({"data": new_patients})[:10])
print("ROC AUC:", round(roc_auc_score(test["cardio"], probabilities), 3))

# %%
# Reading the coefficients
# ------------------------
#
# The fitted model is linear, so each coefficient says how a column pushes the
# predicted log-odds up or down. The blood pressure dominates; the spline
# terms are smaller individually, and they only make sense together, as we'll
# see below.
#
# The column names come from the model itself, which guarantees that they line
# up with the coefficients.

model = learner.find_fitted_estimator("model")
coefs = pd.Series(model.coef_[0], index=model.feature_names_in_)

ax = coefs.sort_values(key=abs).plot(kind="barh")
ax.set(xlabel="coefficient")
plt.show()

# %%
# The effect of the BMI
# ---------------------
#
# The spline coefficients are hard to read one by one; the curve they draw is
# what matters. To draw it we leave the test patients as they are and vary
# their BMI, adjusting each weight to reach a given value, then average the
# predicted risk. This is a partial dependence: everything else stays at its
# real value while the BMI sweeps its range.


def mean_risk_at(bmi_value):
    weight = bmi_value * (test["height"] / 100) ** 2
    return learner.predict_proba({"data": test.assign(weight=weight)})[:, 1].mean()


grid = np.linspace(15, 45, 50)

fig, ax = plt.subplots()
ax.plot(grid, [mean_risk_at(value) for value in grid])
ax.set(xlabel="BMI", ylabel="predicted probability")
plt.show()

# %%
# The risk rises steeply at first and then flattens: extra weight matters most
# for patients who are not already overweight. Nothing in the code computes
# this shape -- it is what the spline basis learned from the training data --
# and yet the model is still a logistic regression whose coefficients we can
# read.
