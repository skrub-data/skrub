"""
.. currentmodule:: skrub

.. _example_subsampling:

Subsampling for faster development
==================================

Here we show how to use :meth:`.skb.subsample_previews()
<Expr.skb.subsample_previews>` to speed-up interactive creation of skrub
expressions by subsampling the data when computing preview results.
"""

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub
import skrub.datasets

dataset = skrub.datasets.fetch_employee_salaries().employee_salaries

full_data = skrub.var("data", dataset)
full_data

# %%
# We are working with a dataset of over 9K rows. As we build up our pipeline,
# we see previews of the intermediate results so we can check that it behaves
# as we expect. However, if some estimators are slow, fitting them and
# computing results on the whole data can slow us down.
#
# So we can tell skrub to subsample the data when computing the previews, with
# :meth:`.skb.subsample_previews() <Expr.skb.subsample_previews>`.

# %%
data = full_data.skb.subsample_previews(n=100)
data

# %%
# The rest of the pipeline will now use only 100 points for its previews.
#
# .. note::
#
#    By default subsampling is applied *only for previews*: the results
#    shown when we display the expression, and the output of calling
#    :meth:`.skb.preview() <Expr.skb.preview>`. For other methods such as
#    :meth:`.skb.get_pipeline() <Expr.skb.get_pipeline>` or
#    :meth:`.skb.cross_validate() <Expr.skb.cross_validate>`, *no subsampling is
#    done by default*. We can explicitly ask for it with ``keep_subsampling=True``
#    as we will see below.
#
# To finish our pipeline we simply apply a TableVectorizer then gradient boosting:

# %%
employees = data.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
salaries = data["current_annual_salary"].skb.mark_as_y()

predictions = employees.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(), y=salaries
)

# %%
# When we display our ``predictions`` expression, we see that the preview is
# computed on a subsample: the result column has only 100 entries.

# %%
predictions

# %%
# We can also turn on subsampling for other methods of the expression, such as
# :meth:`.skb.cross_validate() <Expr.skb.cross_validate>`. Here we run the
# cross-validation on the small subsample of 100 rows we configured. With such
# a small subsample the scores will be very low but this might help us quickly
# detect errors in our cross-validation scheme.

# %%
predictions.skb.cross_validate(keep_subsampling=True)

# %%
# By default, when we do not explicitly ask for ``keep_subsampling=True``, no
# subsampling takes place. Here we run the cross-validation on the full data.
# Note the longer ``fit_time`` and much better ``test_score``.

# %%
predictions.skb.cross_validate()
