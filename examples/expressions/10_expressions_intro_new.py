"""
.. _example_expressions_intro:


Building powerful machine learning pipelines with the skrub DataOps
=========================

In this example, we show how we can use skrub's DataOps to build a machine learning
pipeline that pre-processes data, trains a model, and allows for hyperparameter tuning
on a simple dataset. We will also show how to save the model, load it back,
and then use it to make predictions on new data.

"""

# %%
# We begin by loading the employee salaries dataset, which is a regression dataset
# that contains information about employees and their current annual salaries.
# By default, the ``fetch_employee_salaries`` function returns the training set.
# We will load the test set later, to evaluate our model on unseen data.

from skrub.datasets import fetch_employee_salaries

full_data = fetch_employee_salaries().employee_salaries
training_data = full_data[:8000]
unseen_data = full_data[8000:]

# %%
# We can take a look at the dataset using the `TableReport`.
# This dataset contains numerical, categorical, and datetime features. The column
# `current_annual_salary` is the target variable we want to predict.
#

from skrub import TableReport

TableReport(training_data)
# %%
# Our goal is to predict the `current_annual_salary` of employees based on their
# other features. We will use skrub's DataOps to combine both skrub and scikit-learn
# objects into a single DataOps plan, which will allow us to preprocess the data,
# train a model, and tune hyperparameters.
#
# We begin by defining a skrub variable, which is the entry point for our DataOps plan.
# Additionally, we use the subsample function to limit the number of rows that
# are used to produce previews of each step in the plan. This is useful to speed up
# the development process. At training time, the DataOps will automatically
# use the full dataset.

import skrub

data_var = skrub.var("data", training_data).skb.subsample()

# %%
# Next, we define the starting feature matrix ``X`` and the target variable ``y``.
# We will use the `skb.mark_as_X` and `skb.mark_as_y` methods to mark these variables
# in the DataOps plan. This allows skrub to properly split these objects into
# training and validation steps when executing cross-validation or hyperparameter
# tuning.

X = data_var.drop("current_annual_salary", axis=1).skb.mark_as_X()
y = data_var["current_annual_salary"].skb.mark_as_y()
# %%
# Our first step is to vectorize the features in ``X``. We will use the
# `TableVectorizer` to convert the categorical and numerical features into a
# numerical format that can be used by machine learning algorithms.
# We apply the vectorizer to ``X`` using the `skb.apply` method, which allows us to
# apply any scikit-learn compatible transformer to the skrub variable.

from skrub import TableVectorizer

vectorizer = TableVectorizer()

X_vec = X.skb.apply(vectorizer)
X_vec
# %%
# By clicking on ``Show graph``, we can see the DataOps plan that has been created:
# the plan shows the steps that have been applied to the data so far.
# Now that we have the vectorized features, we can proceed to train a model.
# We use a scikit-learn `HistGradientBoostingRegressor` to predict the target variable.
# We apply the model to the vectorized features using ``.skb.apply``, and pass
# ``y`` as the target variable.
# Note that the resulting ``predictor`` will show the prediction results on the
# preview subsample, but the actual model has not been fitted yet.

from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor()

predictor = X_vec.skb.apply(hgb, y=y)
predictor

# %%
# Now that we have built our entire plan, we can have explore it in more detail
# with the ``.skb.full_report()`` method. This method evaluates each step in
# the plan and shows detailed information about the operations that are being performed.
# The full plan is shown in a browser window and saved on disk.
predictor.skb.full_report()

# %%
# Now that we have defined the predictor, we can create a ``learner``, a
# standalone object that contains all the steps in the DataOps plan. We fit the
# learner, so that it can be used to make predictions on new data.

trained_learner = predictor.skb.get_pipeline(fitted=True)

# %%
# A big advantage of the learner is that it can be pickled and saved to disk,
# allowing us to reuse the trained model later without needing to retrain it.
# The learner contains all steps in the DataOps plan, including the fitted
# vectorizer and the trained model. We can save it using Python's `pickle` module:
# here we use `pickle.dumps` to serialize the learner object into a byte string.

import pickle

saved_model = pickle.dumps(trained_learner)

# %%
# We can now load the saved model back into memory using `pickle.loads`.
loaded_model = pickle.loads(saved_model)

# %%
# Now, we can make predictions on new data using the loaded model, by passing
# a dictionary with the skrub variable names as keys.
# We don't have to create a new variable, as this will be done internally by the
# learner.
# In fact, the ``learner`` is similar to a scikit-learn estimator, but rather
# than taking ``X`` and ``y`` as inputs, it takes a dictionary (the "environment"),
# where each key is the name of one of the skrub variables in the plan.
#
# We can now get the test set of the employee salaries dataset:
unseen_data = fetch_employee_salaries()
unseen_data = unseen_data.employee_salaries[8000:]

# %%
# Then, we can use the loaded model to make predictions on the unseen data by
# passing the environment as dictionary.

predicted_values = loaded_model.predict({"data": unseen_data})
predicted_values

# %%
# We can also evaluate the model's performance using the `score` method, which
# uses the scikit-learn scoring function used by the predictor:
loaded_model.score({"data": unseen_data})

# %%
# So far, we have seen how to build a simple machine learning pipeline using skrub's
# DataOps. However, a major part of optimizing a machine learning model is
# hyperparameter tuning. Skrub's DataOps allow us to easily add hyperparameters
# right where we define the model, by using one of the `choose_*` functions.
# Here, we will use `choose_float` to add a learning rate hyperparameter
# to the `HistGradientBoostingRegressor`.

hgb = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
)
predictor = X_vec.skb.apply(hgb, y=y)
predictor

# %%
# Since now we have can tune the learning rate, we can use the `get_randomized_search`
# method to perform hyperparameter tuning. This method will automatically
# create a randomized search over the hyperparameters defined in the DataOps plan.

search = predictor.skb.get_randomized_search(
    n_iter=4,
    n_jobs=4,
    random_state=0,
    scoring="neg_mean_absolute_error",
    fitted=True,
)

# %%
# We can print the results of the search, along with various hyperparameter
# configurations. In this case, we only have one hyperparameter: the learning rate.

search.results_
# %%
# Another powerful tool to find the best hyperparameters provided by the skrub
# DataOps is the parallel coordinates plot. This plot allows to visualize each
# hyperparameter configuration as a line that connects the hyperparameter values
# to the corresponding score.

search.plot_results()
# %%
# Finally, we can retrieve the best learner from the search results, and save it
# to disk. This learner will contain the best hyperparameter configuration
# found during the search, and can be used to make predictions on new data.

best_learner = search.best_pipeline_
saved_model = pickle.dumps(best_learner)

# %%
# In this example, we have briefly introduced the skrub DataOps, and how they can
# be used to build powerful machine learning pipelines. We have seen how to preprocess
# data, train a model, and tune hyperparameters using skrub's DataOps. We have also
# shown how to save and load the trained model, and how to make predictions on new
# data using the trained model.
#
# However, skrub DataOps are significantly more powerful than what we have shown here.
# The following examples cover specific use cases, such as multi-table machine
# learning, complex hyperparameter tuning, and sampling, in more detail.
