"""
.. _example_subsampling:

Subsampling for faster development
==================================

Here we show how to use :meth:`.skb.subsample() <DataOp.skb.subsample>` to speed up
interactive construction of a skrub DataOps plan by computing previews on a subsampled
version of the original data.

.. currentmodule:: skrub

"""

# %%

import skrub
import skrub.datasets

dataset = skrub.datasets.fetch_employee_salaries().employee_salaries

full_data = skrub.var("data", dataset)
full_data

# %%
# We are working with a dataset of over 9K rows. As we build up our plan,
# we see previews of the intermediate results so we can check that it behaves
# as expected. However, if some estimators are slow, fitting them and
# computing results on the whole data can slow us down.
#
# Lightweight construction of the DataOps plan on a subsample
# ----------------------------------------------------------
#
# We can tell skrub to subsample the data when computing the previews with
# :meth:`.skb.subsample() <DataOp.skb.subsample>`.

# %%
data = full_data.skb.subsample(n=100)
data

# %%
# The rest of the plan will now use only 100 points for its previews.
#
# .. topic:: Subsampling only applies to previews by default
#
#    By default subsampling is applied *only for previews*: the results
#    shown when we display the plan, and the output of calling
#    :meth:`.skb.preview() <DataOp.skb.preview>`. For other methods such as
#    :meth:`.skb.get_learner() <DataOp.skb.get_learner>` or
#    :meth:`.skb.cross_validate() <DataOp.skb.cross_validate>`, *no subsampling is
#    done by default*. We can explicitly ask for it with ``keep_subsampling=True``
#    as we will see below. Even when ``keep_subsampling=True``, subsampling is
#    not applied to the ``predict`` method.
#
# To continue building our plan, we now define X and y:

# %%
employees = data.drop(
    columns="current_annual_salary",
    errors="ignore",
).skb.mark_as_X()

salaries = data["current_annual_salary"].skb.mark_as_y()

# %%
# And finally we apply a TableVectorizer then gradient boosting:

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

predictions = employees.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(), y=salaries
)

# %%
#
# All the lines above run very fast, including fitting the predictor above.
#
# When we display our ``predictions`` DataOp, we see that the preview is
# computed on a subsample: the result column has only 100 entries.

# %%
predictions

# %%
# We can also turn on subsampling for other DataOps methods, such as
# :meth:`.skb.cross_validate() <DataOp.skb.cross_validate>`. Here we run the
# cross-validation on the small subsample of 100 rows we configured. With such
# a small subsample the scores will be very low but this might help us quickly
# detect errors in our cross-validation scheme.

# %%
predictions.skb.cross_validate(keep_subsampling=True)

# %%
# Evaluating the DataOps plan on the full data
# -------------------------------------------
# By default, when we do not explicitly ask for ``keep_subsampling=True``, no
# subsampling takes place.
#
# Here we run the cross-validation **on the full data**.
# Note the longer ``fit_time`` and much better ``test_score``.

# %%
predictions.skb.cross_validate()
