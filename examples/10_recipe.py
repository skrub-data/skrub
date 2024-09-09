"""
Using the Recipe
================

Skrub provides the |Recipe|, a convenient and interactive way to create
machine-learning models for tabular data.

The |Recipe| is a helper to build a scikit-learn |Pipeline| (a chain of data
processing steps) and its ranges of hyperparameters (the values that control
the behavior of each step in the pipeline, such as the number of trees in a
random forest).

The |Recipe| brings 3 major benefits:

- The data processing pipeline can be built step by step, while easily checking
  the output of data transformations we have added so far. This makes
  development faster and more interactive.

- When we add a transformation, we can easily specify which columns it should
  modify. Tabular data is heterogeneous and many processing steps only apply to
  some of the columns in our dataframe.

- When we add a processing step, we can specify a range for any of its
  parameters, rather than a single value. Thus the ranges of hyperparameters to
  consider for tuning are provided directly inside the estimator that uses
  them. Moreover, these hyperparameters can be given human-readable names which
  are useful for inspecting hyperparameter search results.

.. admonition:: The Recipe has no ``fit()`` or ``predict()``

    The recipe is not a scikit-learn estimator with ``fit`` and ``predict``
    methods. Rather, it is a tool for configuring a scikit-learn estimator.
    Once we have created a Recipe, we must call one of its methods such as
    ``get_pipeline``, ``get_grid_search``, or ``get_randomized_search`` to
    obtain the corresponding scikit-learn object.

    For example:

    - ``get_pipeline`` returns a scikit-learn |Pipeline| which applies all the
      steps we have configured without any hyperparameter tuning.
    - ``get_grid_search`` returns a scikit-learn |GridSearchCV| which uses the same
      pipeline but runs a nested cross-validation loop to evaluate different
      combinations of hyperparameters from the ranges we have specified and select
      the best configuration.

.. |Pipeline| replace::
    :class:`~sklearn.pipeline.Pipeline`

.. |GridSearchCV| replace::
    :class:`~sklearn.model_selection.GridSearchCV`

.. |HGB| replace::
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`

.. |TargetEncoder| replace::
    :class:`~sklearn.preprocessing.TargetEncoder`

.. |Recipe| replace::
     :class:`~skrub.Recipe`

.. |TableReport| replace::
     :class:`~skrub.TableReport`

.. |DatetimeEncoder| replace::
     :class:`~skrub.DatetimeEncoder`

.. |MinHashEncoder| replace::
     :class:`~skrub.MinHashEncoder`

"""


# %%
# Getting a preview of the transformed data
# -----------------------------------------
#
# The |Recipe| can be initialized with the full dataset (including the
# prediction target, ``y``). We can tell it which columns constitute the target
# and should be kept separate from the rest.

# %%
from skrub import Recipe, datasets

dataset = datasets.fetch_employee_salaries()
df = dataset.X
df["salary"] = dataset.y

recipe = Recipe(df, y_cols="salary", n_jobs=8)
recipe

# %%
# Our recipe does not contain any transformations yet, except for
# the built-in one which separates the target columns from the predictive
# features.

# %%
# We can use ``sample()`` to get a sample of the data transformed
# by the steps we have added to the recipe so far. The recipe draws a
# random sample from the dataframe we used to initialize it, and applies the
# transformations we have specified.
#
# At this point we have not added any transformations yet, so we just get a
# sample from the original dataframe.

# %%
recipe.sample()

# %%
# If instead of a random sample we want to see the transformation of the first
# few rows in their original order, we can use ``head()`` instead of ``sample()``.
#
# We can ask for a |TableReport| of the transformed sample, to inspect it
# more easily:

# %%
recipe.get_report()

# %%
# Adding transformations to specific columns
# ------------------------------------------
#
# We can use the report above to explore the dataset and plan the
# transformations we need to apply to the different columns.
# (Note that in the "Distributions" tab, we can select columns and construct a
# list of column names that we can copy-paste to save some typing.)
#
# We notice that the ``date_first_hired`` column contains dates such as
# ``'01/21/2001'`` but has the dtype ``ObjectDType`` -- which is the pandas
# representation of strings. Our dates are represented as strings, which we
# need to parse and transform into proper datetime objects from which we
# can then extract information.
#
# Without Skrub, we might do that on the whole dataframe using something like:

# %%

# Don't do this:
# df["date_first_hired"] = pd.to_datetime(df["date_first_hired"])

# %%
# However, we would then need to remember to apply the same transformation when
# we receive new data for which we are asked to make a prediction. And we would
# also need to store the detected datetime format to be sure we apply the same
# transformation. Skrub helps us integrate that data-wrangling operation into
# our machine-learning model.
#
# To do so, we now add the first step to our recipe.

# %%
from skrub import ToDatetime
from skrub import selectors as s

recipe = recipe.add(ToDatetime(), cols="date_first_hired")
recipe

# %%
# To add a transformation, we call ``recipe.add()`` with the transformer to
# apply and optionally ``cols``, the set of columns it should modify.
#
# The columns can be a single column name (as shown above), or a list of column names.
# Skrub also provides more powerful column selectors as we will see below.
#
# .. admonition:: ``recipe.add()`` returns a new object
#
#     Note that we wrote ``recipe = recipe.add(...)``. This is necessary because
#     ``add()`` does not modify the recipe in-place. Instead, it returns a new
#     |Recipe| object which contains the added step. To keep working with this new,
#     augmented recipe, we bind the variable ``recipe`` to the new object.
#
# We can check what the data looks like with the added transformation. In the
# report's **"Show: All columns"** dropdown, we can filter which columns we want
# to see in the report. The default is to show all of them, but if we select
# **"Modified by last step"**, we see only those that were created or modified by
# the transformation we just added. In our case, this is the
# ``"date_first_hired"`` column. We can see that it has a new dtype: it is now
# a true datetime column.

# %%
recipe.get_report()

# %%
# More flexible column selection
# ------------------------------
#
# Now that we have converted ``"date_first_hired"`` to datetimes, we can extract
# temporal features such as the year and month with Skrub's |DatetimeEncoder|.
#
# We could again specify ``cols="date_first_hired"`` but we will take this
# opportunity to introduce more flexible ways of selecting columns.
#
# The ``skrub.selectors`` module provides objects that we can use instead of
# an explicit list of column names, and which allow us to select columns based
# on their type, patterns in their name, etc. If you have used Polars or Ibis
# selectors before, they work in the same way.
#
# For example, ``skrub.selectors.any_date()`` selects all columns that have a
# datetime or date dtype.

# %%
from skrub import DatetimeEncoder

recipe = recipe.add(DatetimeEncoder(), cols=s.any_date())
recipe

# %%
# If we look at the preview, we see that ``any_date()`` has selected the
# ``"date_first_hired"`` column and that the |DatetimeEncoder| has extracted
# the year, month, day, and total number of seconds since the Unix epoch (start
# of 1970). You can easily check this with **"Show: Modified by last step"**.

# %%
recipe.get_report()

# %%
# Selectors can be combined with the same operators as `Python sets
# <https://docs.python.org/3/tutorial/datastructures.html#sets>`_. For example,
# ``s.numeric() - "ID"`` would select all numeric columns except "ID", or
# ``s.categorical() | s.string()`` would get all columns that have either a
# Categorical or a string dtype.
#
# Here we use ``s.string() & s.cardinality_below(30)`` to select columns that
# contain strings *and* have a cardinality (number of unique values) strictly
# below 30. Those string columns with few unique values probably represent
# categories, so we will convert them to an actual ``Categorical`` dtype so
# that the rest of the pipeline can recognize and process them as such.
# High-cardinality string columns will be dealt with separately.


# %%
from skrub import ToCategorical

recipe = recipe.add(ToCategorical(), cols=s.string() & s.cardinality_below(30))
recipe

# %%
# Specifying alternative transformations
# --------------------------------------
# The last columns in our dataframe we have not handled yet are the string
# columns with many unique values. They are those still left with the string
# dtype: ``"division"`` and ``"employee_position_title"``.
#
# Treating those as categorical columns is probably not a good choice.
# Due to the high number of unique values, many categories would have very few
# examples which would make learning difficult.
#
# For such cases, the |TargetEncoder| from scikit-learn is often a good choice.
# It represents a string or category by the average of the target ``y`` among
# samples of the same category in the training set. For example, ``"Bus
# Operator"`` would be replaced by the average salary of bus operators in the
# training set.
#
# While the |TargetEncoder| is often effective, it treats each unique value
# independently. To the |TargetEncoder|, ``"Police officer II"`` and ``"Police
# officer III"`` are just 2 distinct values with no special relation. However,
# we can see that these 2 entries for ``"employee_position_title"`` have very
# similar representations ("surface forms") and that could be useful
# information for our machine-learning model.
#
# To use this information, the |MinHashEncoder| from Skrub can be very
# effective. It creates features that capture the presence of substrings
# ("n-grams", such as ``"Pol"``) in the input.
#
# Both options seem reasonable, so we would like to try both and keep the one
# that works the best. Instead of a single estimator, the |Recipe| allows us to
# add a choice: several options from which the best-performing one will be
# kept.

# %%
from sklearn.preprocessing import TargetEncoder

from skrub import MinHashEncoder, choose_from

recipe = recipe.add(
    choose_from(
        {"target": TargetEncoder(), "minhash": MinHashEncoder()}, name="encoder"
    ),
    cols=s.string(),
)
recipe

# %%
# Here we told the recipe that for this processing step, it should choose from
# 2 possibilities: either a |TargetEncoder| or a |MinHashEncoder|. The keys in
# the dictionary (``"target"`` and ``"minhash"`` respectively) are
# human-readable labels that the recipe will use when displaying results. If we
# do not want to specify such labels we can also pass a list instead of a
# dictionary: we could have written ``choose_from([TargetEncoder(),
# MinHashEncoder()])``.
#
# When we ask for a sample or a report, the recipe will apply the default choice
# which is the first one -- in our case, the |TargetEncoder|.


# %%
# Specifying hyperparameter ranges
# --------------------------------
# We are now done with preprocessing. If we display another report, we see that
# all columns are now encoded as numbers or Categorical values, which the
# supervised learner we are about to add can handle.
#
# We can add the final step in our pipeline: the supervised learner which will
# received the extracted or preprocessed features and learn to predict an
# employee's salary.
#
# We will use a |HGB| from scikit-learn, which works very well for tabular data
# and can handle categorical columns (provided they have a Categorical dtype,
# which we made sure is the case) and missing values.
#
# The |HGB| has a few hyperparameters we can set; an important one is the
# learning rate. By default, scikit-learn will use ``learning_rate=0.1``. As it
# can have an impact on the fitting time and on the quality of predictions, we
# want to try a few values and pick the best. Here also, the recipe allows us
# to provide a choice instead of a single value.
#
# We could rely on `choose_from` as before, to pick from a few possibilities.
# But as the learning rate is a number, we can use a more specialized kind of
# choice: ``choose_float``. It allows us to specify the start and end of a
# range, and whether values should be sampled uniformly or on a log scale in
# this range.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

from skrub import choose_float

recipe = recipe.add(
    HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        learning_rate=choose_float(0.001, 1.0, log=True, name="learning rate"),
    )
)
recipe

# %%
# Note that here the choice is not among several estimators, but for the
# parameter of the |HGB|. We pass the choice directly in the estimator itself,
# where we would normally give a value for the parameter. The ``"learning
# rate"`` string is an optional human-readable label for this particular
# choice.
#
# Choices can be nested (as deep as we want): we can have a choice between
# estimators, where each of the estimators contains choices in its parameters.
#
# If instead of a floating-point value we needed an integer (e.g. for a number
# of dimensions or iterations), we could use ``choose_int``.


# %%
# Obtaining and fitting our machine-learning model
# ------------------------------------------------
# We are now done building our |Recipe|, and we can try it on some data and see
# how it performs.
#
# As mentioned at the top of the page, a |Recipe| is not a scikit-learn
# estimator with the usual ``fit`` and ``transform`` methods. Rather, it is a
# way to configure such an object.
# We must call one of the recipe's method for getting the scikit-learn object
# we have configured.
#
# We can use ``recipe.get_pipeline()`` which will return a scikit-learn
# |Pipeline|. This object applies all the transformation steps we have added
# and the final supervised learner. However, it performs no hyperparameter
# selection. Wherever we specified a choice, it will just use the default value
# (the first option for ``choose_from``, or the middle of the range for
# ``choose_float`` and ``choose_int``).
#
# To perform hyperparameter selection, we can use ``recipe.get_grid_search()``
# (returns a |GridSearchCV|) or ``recipe.get_randomized_search()`` (returns a
# |RandomizedSearchCV|). Both run a nested loop of cross-validation on their
# training data to select the best hyperparameters, then refit the best
# combination of hyperparameters on the whole training data. The difference is
# that the grid search will try all possible combinations, whereas the
# randomized search will sample a few possible combinations (we decide how
# many).
#
# See the scikit-learn `documentation
# <https://scikit-learn.org/stable/modules/grid_search.html>`_ for more
# information about hyperparameter search.
#
# Here we have specified a continuous range of values for the learning rate, so
# "all possible values" does not make sense and we will sample a few instead:
# we use the randomized search.

# %%
from sklearn.metrics import r2_score

randomized_search = recipe.get_randomized_search(n_iter=10, cv=3, verbose=1)
randomized_search.fit(recipe.get_x_train(), recipe.get_y_train())

predictions = randomized_search.predict(recipe.get_x_test())
score = r2_score(recipe.get_y_test(), predictions)
print(f"R² score: {score:.2f}")

# %%
# We can use ``get_x_train()``, ``get_x_test()`` to get a default train/test
# split of our data. If we wanted the whole data instead, for example to run a
# cross-validation loop with ``sklearn.model_selection.cross_validate``, we
# could use ``get_x()`` and ``get_y()``.
#

# %%
# Inspecting hyperparameter search results
# ----------------------------------------
# Here we fitted a single model instead of running a cross-validation so that
# we could inspect the results stored during the hyperparameter search
# procedure. By default, the |RandomizedSearchCV| can provide those results as
# a dictionary with somewhat complicated keys.
#
# We can ask the |Recipe| to create a nicer display of those results, using the
# human-readable labels we provided earlier for the different hyperparameter
# choices.

# %%
recipe.get_cv_results_table(randomized_search)

# %%
# If we have the ``plotly`` library installed, we can have a more visual and
# interactive exploration of the hyperparameter search results.

# %%
recipe.plot_parallel_coord(randomized_search)

# %%
# In the interactive figure above, each line represents a combination of
# hyperparameters. It crosses each vertical bar at the height which corresponds
# to this combination. So if we start from the top-left of the plot, we can see
# the score of the best model, and then follow the line to see what was the
# selected encoder, then learning rate, and finally the fitting and prediction
# times for this particular hyperparameter combination.
#
# If we click and drag vertically on one of the vertical lines, we can select a
# range for the corresponding value and the combinations that fall outside of
# that range are greyed out. This allows us (for example) to highlight only the
# best-performing models.
#
# By doing so, we realize that both min-hash and target encoding can perform
# well but target encoding is faster, and that the best learning rates tend to
# be towards the middle or the high end of the range we specified.
#
# See the `Plotly documentation
# <https://plotly.com/python/parallel-coordinates-plot/>`_ for more about those
# plots and screen recordings of the possible interactions.
