.. _changes:

========
Changes
========

.. currentmodule:: skrub

Ongoing development
===================

Skrub is a very recent package.
It is currently undergoing fast development and backward compatibility is not ensured.

Major changes
-------------

Minor changes
-------------

Release 0.2.0
=============

Major changes
-------------
* The :class:`Joiner` has been adapted to support polars dataframes. :pr:`945` by :user:`Théo Jolivet <TheooJ>`.

* The :class:`TableVectorizer` now consistently applies the same transformation
  across different calls to `transform`. There also have been some breaking
  changes to its functionality: (i) all transformations are now applied
  independently to each column, i.e. it does not perform multivariate
  transformations (ii) in ``specific_transformers`` the same column may not be
  used twice (go through 2 different transformers).
  :pr:`902` by :user:`Jérôme Dockès <jeromedockes>`.

* Some parameters of :class:`TableVectorizer` have been renamed:
  `high_cardinality_transformer` → `high_cardinality`,
  `low_cardinality_transformer` → `low_cardinality`,
  `datetime_transformer` → `datetime`, `numeric_transformer` → `numeric`.
  :pr:`947` by :user:`Jérôme Dockès <jeromedockes>`.

* The :class:`GapEncoder` and :class:`MinHashEncoder` are now a single-column
  transformers: their ``fit``, ``fit_transform`` and ``transform`` methods
  accept a single column (a pandas or polars Series). Dataframes and numpy
  arrays are not accepted.
  :pr:`920` and :pr:`923` by :user:`Jérôme Dockès <jeromedockes>`.

* Added the :class:`MultiAggJoiner` that allows to augment a main table with
  multiple auxiliary tables. :pr:`876` by :user:`Théo Jolivet <TheooJ>`.

* :class:`AggJoiner` now only accepts a single table as an input, and some of its
  parameters were renamed to be consistent with the :class:`MultiAggJoiner`.
  It now has a ``key``` parameter that allows to join main and auxiliary tables that share
  the same column names. :pr:`876` by :user:`Théo Jolivet <TheooJ>`.

* :func:`tabular_learner` has been added to easily create a supervised
  learner that works well on tabular data. :pr:`926` by :user:`Jérôme Dockès
  <jeromedockes>`.

Minor changes
-------------

* :class:`GapEncoder` and :class:`MinHashEncoder` used to modify their input
  in-place, replacing missing values with a string. They no longer do so. Their
  parameter `handle_missing` has been removed; now missing values are always
  treated as the empty string.
  :pr:`930` by :user:`Jérôme Dockès <jeromedockes>`.

* The minimum supported python version is now 3.9
  :pr:`939` by :user:`Jérôme Dockès <jeromedockes>`.

* Skrub supports numpy 2. :pr:`946` by :user:`Jérôme Dockès <jeromedockes>`.

* :func:`~datasets.fetch_ken_embeddings` now add suffix even with the default
  value for the parameter `pca_components`.
  :pr:`956` by :user:`Guillaume Lemaitre <glemaitre>`.

* :class:`Joiner` now performs some preprocessing (the same as done by the
  :class:`TableVectorizer`, eg trying to parse dates, converting pandas object
  columns with mixed types to a single type) on the joining columns before
  vectorizing them. :pr:`972` by :user:`Jérôme Dockès <jeromedockes>`.

skrub release 0.1.1
===================

This is a bugfix release to adapt to the most recent versions of pandas (2.2) and
scikit-learn (1.5). There are no major changes to the functionality of skrub.


skrub release 0.1.0
===================


Major changes
-------------
* :class:`TargetEncoder` has been removed in favor of
  :class:`sklearn.preprocessing.TargetEncoder`, available since scikit-learn 1.3.

* :class:`Joiner` and :func:`fuzzy_join` support several ways of rescaling
  distances; ``match_score`` has been replaced by ``max_dist``; bugs which
  prevented the Joiner to consistently vectorize inputs and accept or reject
  matches across calls to transform have been fixed. :pr:`821` by :user:`Jérôme
  Dockès <jeromedockes>`.

* :class:`InterpolationJoiner` was added to join two tables by using
  machine-learning to infer the matching rows from the second table.
  :pr:`742` by :user:`Jérôme Dockès <jeromedockes>`.

* Pipelines including :class:`TableVectorizer` can now be grid-searched, since
  we can now call `set_params` on the default transformers of :class:`TableVectorizer`.
  :pr:`814` by :user:`Vincent Maladiere <Vincent-Maladiere>`

* :func:`to_datetime` is now available to support pandas.to_datetime
  over dataframes and 2d arrays.
  :pr:`784` by :user:`Vincent Maladiere <Vincent-Maladiere>`

* Some parameters of :class:`Joiner` have changed. The goal is to harmonize
  parameters across all estimator that perform join(-like) operations, as
  discussed in `#751 <https://github.com/skrub-data/skrub/discussions/751>`_.
  :pr:`757` by :user:`Jérôme Dockès <jeromedockes>`.

* :func:`dataframe.pd_join`, :func:`dataframe.pd_aggregate`,
  :func:`dataframe.pl_join` and :func:`dataframe.pl_aggregate`
  are now available in the dataframe submodule.
  :pr:`733` by :user:`Vincent Maladiere <Vincent-Maladiere>`

* :class:`FeatureAugmenter` is renamed to :class:`Joiner`.
  :pr:`674` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* :func:`fuzzy_join` and :class:`FeatureAugmenter` can now join on datetime columns.
  :pr:`552` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* :class:`Joiner` now supports joining on multiple column keys.
  :pr:`674` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* The signatures of all encoders and functions have been revised to enforce
  cleaner calls. This means that some arguments that could previously be passed
  positionally now have to be passed as keywords.
  :pr:`514` by :user:`Lilian Boulard <LilianBoulard>`.

* Parallelized the :class:`GapEncoder` column-wise. Parameters `n_jobs` and `verbose`
  added to the signature. :pr:`582` by :user:`Lilian Boulard <LilianBoulard>`

* Introducing :class:`AggJoiner`, a transformer performing
  aggregation on auxiliary tables followed by left-joining on a base table.
  :pr:`600` by :user:`Vincent Maladiere <Vincent-Maladiere>`.

* Introducing :class:`AggTarget`, a transformer performing
  aggregation on the target y, followed by left-joining on a base table.
  :pr:`600` by :user:`Vincent Maladiere <Vincent-Maladiere>`.

* Added the :class:`SelectCols` and :class:`DropCols` transformers that allow
  selecting a subset of a dataframe's columns inside of a pipeline. :pr:`804` by
  :user:`Jérôme Dockès <jeromedockes>`.


Minor changes
-------------
* :class:`DatetimeEncoder` doesn't remove constant features anymore.
  It also supports an 'errors' argument to raise or coerce errors during
  transform, and a 'add_total_seconds' argument to include the number of
  seconds since Epoch.
  :pr:`784` by :user:`Vincent Maladiere <Vincent-Maladiere>`

* Scaling of ``matching_score`` in :func:`fuzzy_join` is now between 0 and 1; it used to be between 0.5 and 1. Moreover, the division by 0 error that occurred when all rows had a perfect match has been fixed. :pr:`802` by :user:`Jérôme Dockès <jeromedockes>`.

* :class:`TableVectorizer` is now able to apply parallelism at the column level rather than the transformer level. This is the default for univariate transformers, like :class:`MinHashEncoder`, and :class:`GapEncoder`.
  :pr:`592` by :user:`Leo Grinsztajn <LeoGrin>`

* ``inverse_transform`` in :class:`SimilarityEncoder` now works as expected; it used to raise an exception. :pr:`801` by :user:`Jérôme Dockès <jeromedockes>`.

* :class:`TableVectorizer` propagate the `n_jobs` parameter to the underlying
  transformers except if the underlying transformer already set explicitly `n_jobs`.
  :pr:`761` by :user:`Leo Grinsztajn <LeoGrin>`, :user:`Guillaume Lemaitre <glemaitre>`,
  and :user:`Jerome Dockes <jeromedockes>`.


* Parallelized the :func:`deduplicate` function. Parameter `n_jobs`
  added to the signature. :pr:`618` by :user:`Jovan Stojanovic <jovan-stojanovic>`
  and :user:`Lilian Boulard <LilianBoulard>`

* Functions :func:`datasets.fetch_ken_embeddings`, :func:`datasets.fetch_ken_table_aliases`
  and :func:`datasets.fetch_ken_types` have been renamed.
  :pr:`602` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* Make `pyarrow` an optional dependencies to facilitate the integration
  with `pyodide`.
  :pr:`639` by :user:`Guillaume Lemaitre <glemaitre>`.

* Bumped minimal required Python version to 3.10. :pr:`606` by
  :user:`Gael Varoquaux <GaelVaroquaux>`

* Bumped minimal required versions for the dependencies:
  - numpy >= 1.23.5
  - scipy >= 1.9.3
  - scikit-learn >= 1.2.1
  - pandas >= 1.5.3 :pr:`613` by :user:`Lilian Boulard <LilianBoulard>`

* You can now pass column-specific transformers to :class:`TableVectorizer`
  using the `specific_transformers` argument.
  :pr:`583` by :user:`Lilian Boulard <LilianBoulard>`.

* Do not support 1-D array (and pandas Series) in :class:`TableVectorizer`. Pass a
  2-D array (or a pandas DataFrame) with a single column instead. This change is for
  compliance with the scikit-learn API.
  :pr:`647` by :user:`Guillaume Lemaitre <glemaitre>`

* Fixes a bug in :class:`TableVectorizer` with `remainder`: it is now cloned if it's
  a transformer so that the same instance is not shared between different
  transformers.
  :pr:`678` by :user:`Guillaume Lemaitre <glemaitre>`

* :class:`GapEncoder` speedup :pr:`680` by :user:`Leo Grinsztajn <LeoGrin>`

  - Improved :class:`GapEncoder`'s early stopping logic. The parameters `tol` and `min_iter`
    have been removed. The parameter `max_no_improvement` can now be used to control the
    early stopping.
    :pr:`663` by :user:`Simona Maggio <simonamaggio>`
    :pr:`593` by  :user:`Lilian Boulard <LilianBoulard>`
    :pr:`681` by  :user:`Leo Grinsztajn <LeoGrin>`

  - Implementation improvement leading to a ~x5 speedup for each iteration.

  - Better default hyperparameters: `batch_size` now defaults to 1024, and `max_iter_e_steps`
    to 1.

* Removed the `most_frequent` and `k-means` strategies from the :class:`SimilarityEncoder`.
  These strategy were used for scalability reasons, but we recommend using the :class:`MinHashEncoder`
  or the :class:`GapEncoder` instead. :pr:`596` by :user:`Leo Grinsztajn <LeoGrin>`

* Removed the `similarity` argument from the :class:`SimilarityEncoder` constructor,
  as we only support the ngram similarity. :pr:`596` by :user:`Leo Grinsztajn <LeoGrin>`

* Added the `analyzer` parameter to the :class:`SimilarityEncoder` to allow word counts
  for similarity measures. :pr:`619` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* skrub now uses modern type hints introduced in PEP 585.
  :pr:`609` by :user:`Lilian Boulard <LilianBoulard>`

* Some bug fixes for :class:`TableVectorizer` ( :pr:`579`):

  - `check_is_fitted` now looks at `"transformers_"` rather than `"columns_"`
  - the default of the `remainder` parameter in the docstring is now `"passthrough"`
    instead of `"drop"` to match the implementation.
  - uint8 and int8 dtypes are now considered as numerical columns.

* Removed the leading "<" and trailing ">" symbols from KEN entities
  and types.
  :pr:`601` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* Add `get_feature_names_out` method to :class:`MinHashEncoder`.
  :pr:`616` by :user:`Leo Grinsztajn <LeoGrin>`

* Removed `requests` from the requirements. :pr:`613` by :user:`Lilian Boulard <LilianBoulard>`

* :class:`TableVectorizer` now handles mixed types columns without failing
  by converting them to string before type inference.
  :pr:`623`by :user:`Leo Grinsztajn <LeoGrin>`

* Moved the default storage location of data to the user's home folder.
  :pr:`652` by :user:`Felix Lefebvre <flefebv>` and
  :user:`Gael Varoquaux <GaelVaroquaux>`

* Fixed bug when using :class:`TableVectorizer`'s `transform` method on
  categorical columns with missing values.
  :pr:`644` by :user:`Leo Grinsztajn <LeoGrin>`

* :class:`TableVectorizer` never output a sparse matrix by default. This can be changed by
  increasing the `sparse_threshold` parameter. :pr:`646` by :user:`Leo Grinsztajn <LeoGrin>`

* :class:`TableVectorizer` doesn't fail anymore if an infered type doesn't work during transform.
  The new entries not matching the type are replaced by missing values. :pr:`666` by :user:`Leo Grinsztajn <LeoGrin>`

- Dataset fetcher :func:`datasets.fetch_employee_salaries` now has a parameter
  `overload_job_titles` to allow overloading the job titles
  (`employee_position_title`) with the column `underfilled_job_title`,
  which provides some more information about the job title.
  :pr:`581` by :user:`Lilian Boulard <LilianBoulard>`

* Fix bugs which was triggered when `extract_until` was "year", "month", "microseconds"
  or "nanoseconds", and add the option to set it to `None` to only extract `total_time`,
  the time from epoch. :class:`DatetimeEncoder`. :pr:`743` by :user:`Leo Grinsztajn <LeoGrin>`

Before skrub: dirty_cat
========================

Skrub was born from the `dirty_cat <http://dirty-cat.github.io>`__
package.

Dirty-cat release 0.4.1
==========================

Major changes
-------------
* :func:`fuzzy_join` and :class:`FeatureAugmenter` can now join on numerical columns based on the euclidean distance.
  :pr:`530` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* :func:`fuzzy_join` and :class:`FeatureAugmenter` can perform many-to-many joins on lists of numerical or string key columns.
  :pr:`530` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* :func:`GapEncoder.transform` will not continue fitting of the instance anymore.
  It makes functions that depend on it (:func:`~GapEncoder.get_feature_names_out`,
  :func:`~GapEncoder.score`, etc.) deterministic once fitted.
  :pr:`548` by :user:`Lilian Boulard <LilianBoulard>`

* :func:`fuzzy_join` and :class:`FeatureAugmenter` now perform joins on missing values as in `pandas.merge`
  but raises a warning. :pr:`522` and :pr:`529` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* Added :func:`get_ken_table_aliases` and :func:`get_ken_types` for exploring
  KEN embeddings. :pr:`539` by :user:`Lilian Boulard <LilianBoulard>`.


Minor changes
-------------
* Improvement of date column detection and date format inference in :class:`TableVectorizer`. The
  format inference now tries to find a format which works for all non-missing values of the column, and only
  tries pandas default inference if it fails.
  :pr:`543` by :user:`Leo Grinsztajn <LeoGrin>`
  :pr:`587` by :user:`Leo Grinsztajn <LeoGrin>`



Dirty-cat Release 0.4.0
=========================

Major changes
-------------
* `SuperVectorizer` is renamed as :class:`TableVectorizer`, a warning is raised when using the old name.
  :pr:`484` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* New experimental feature: joining tables using :func:`fuzzy_join` by approximate key matching. Matches are based
  on string similarities and the nearest neighbors matches are found for each category.
  :pr:`291` by :user:`Jovan Stojanovic <jovan-stojanovic>` and :user:`Leo Grinsztajn <LeoGrin>`

* New experimental feature: :class:`FeatureAugmenter`, a transformer
  that augments with :func:`fuzzy_join` the number of features in a main table by using information from auxiliary tables.
  :pr:`409` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* Unnecessary API has been made private: everything (files, functions, classes)
  starting with an underscore shouldn't be imported in your code. :pr:`331` by :user:`Lilian Boulard <LilianBoulard>`

* The :class:`MinHashEncoder` now supports a `n_jobs` parameter to parallelize
  the hashes computation. :pr:`267` by :user:`Leo Grinsztajn <LeoGrin>` and :user:`Lilian Boulard <LilianBoulard>`.

* New experimental feature: deduplicating misspelled categories using :func:`deduplicate` by clustering string distances.
  This function works best when there are significantly more duplicates than underlying categories.
  :pr:`339` by :user:`Moritz Boos <mjboos>`.

Minor changes
-------------
* Add example `Wikipedia embeddings to enrich the data`. :pr:`487` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* **datasets.fetching**: contains a new function :func:`get_ken_embeddings` that can be used to download Wikipedia
  embeddings and filter them by type.

* **datasets.fetching**: contains a new function :func:`fetch_world_bank_indicator` that can be used to download indicators
  from the World Bank Open Data platform.
  :pr:`291` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* Removed example `Fitting scalable, non-linear models on data with dirty categories`. :pr:`386` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* :class:`MinHashEncoder`'s :func:`minhash` method is no longer public. :pr:`379` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* Fetching functions now have an additional argument ``directory``,
  which can be used to specify where to save and load from datasets.
  :pr:`432` by :user:`Lilian Boulard <LilianBoulard>`

* Fetching functions now have an additional argument ``directory``,
  which can be used to specify where to save and load from datasets.
  :pr:`432` and :pr:`453` by :user:`Lilian Boulard <LilianBoulard>`

* The :class:`TableVectorizer`'s default `OneHotEncoder` for low cardinality categorical variables now defaults
  to `handle_unknown="ignore"` instead of `handle_unknown="error"` (for sklearn >= 1.0.0).
  This means that categories seen only at test time will be encoded by a vector of zeroes instead of raising an error. :pr:`473` by :user:`Leo Grinsztajn <LeoGrin>`

Bug fixes
---------

* The :class:`MinHashEncoder` now considers `None` and empty strings as missing values, rather
  than raising an error. :pr:`378` by :user:`Gael Varoquaux <GaelVaroquaux>`

Dirty-cat Release 0.3.0
==========================

Major changes
-------------

* New encoder: :class:`DatetimeEncoder` can transform a datetime column into several numerical columns
  (year, month, day, hour, minute, second, ...). It is now the default transformer used
  in the :class:`TableVectorizer` for datetime columns. :pr:`239` by :user:`Leo Grinsztajn <LeoGrin>`

* The :class:`TableVectorizer` has seen some major improvements and bug fixes:

  - Fixes the automatic casting logic in ``transform``.
  - To avoid dimensionality explosion when a feature has two unique values, the default encoder (:class:`~sklearn.preprocessing.OneHotEncoder`) now drops one of the two vectors (see parameter `drop="if_binary"`).
  - ``fit_transform`` and ``transform`` can now return unencoded features, like the :class:`~sklearn.compose.ColumnTransformer`'s behavior. Previously, a ``RuntimeError`` was raised.

  :pr:`300` by :user:`Lilian Boulard <LilianBoulard>`

* **Backward-incompatible change in the TableVectorizer**:
  To apply ``remainder`` to features (with the ``*_transformer`` parameters),
  the value ``'remainder'`` must be passed, instead of ``None`` in previous versions.
  ``None`` now indicates that we want to use the default transformer. :pr:`303` by :user:`Lilian Boulard <LilianBoulard>`

* Support for Python 3.6 and 3.7 has been dropped. Python >= 3.8 is now required. :pr:`289` by :user:`Lilian Boulard <LilianBoulard>`

* Bumped minimum dependencies:

  - scikit-learn>=0.23
  - scipy>=1.4.0
  - numpy>=1.17.3
  - pandas>=1.2.0 :pr:`299` and :pr:`300` by :user:`Lilian Boulard <LilianBoulard>`

* Dropped support for Jaro, Jaro-Winkler and Levenshtein distances.

  - The :class:`SimilarityEncoder` now exclusively uses ``ngram`` for similarities,
    and the `similarity` parameter is deprecated. It will be removed in 0.5. :pr:`282` by :user:`Lilian Boulard <LilianBoulard>`

Notes
-----

* The ``transformers_`` attribute of the :class:`TableVectorizer` now contains column
  names instead of column indices for the "remainder" columns. :pr:`266` by :user:`Leo Grinsztajn <LeoGrin>`


Dirty-cat Release 0.2.2
=========================

Bug fixes
---------

* Fixed a bug in the :class:`TableVectorizer` causing a :class:`FutureWarning`
  when using the :func:`get_feature_names_out` method. :pr:`262` by :user:`Lilian Boulard <LilianBoulard>`


Dirty-cat Release 0.2.1
==========================

Major changes
-------------

* Improvements to the :class:`TableVectorizer`

    - Type detection works better: handles dates, numerics columns encoded as strings, or numeric columns containing strings for missing values.

  :pr:`238` by :user:`Leo Grinsztajn <LeoGrin>`

* :func:`get_feature_names` becomes :func:`get_feature_names_out`, following changes in the scikit-learn API.
  :func:`get_feature_names` is deprecated in scikit-learn > 1.0. :pr:`241` by :user:`Gael Varoquaux <GaelVaroquaux>`

* Improvements to the :class:`MinHashEncoder`
    - It is now possible to fit multiple columns simultaneously with the :class:`MinHashEncoder`.
      Very useful when using for instance the :func:`~sklearn.compose.make_column_transformer` function,
      on multiple columns.

  :pr:`243` by :user:`Jovan Stojanovic <jovan-stojanovic>`


Bug-fixes
---------

* Fixed a bug that resulted in the :class:`GapEncoder` ignoring the analyzer argument. :pr:`242` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* :class:`GapEncoder`'s `get_feature_names_out` now accepts all iterators, not just lists. :pr:`255` by :user:`Lilian Boulard <LilianBoulard>`

* Fixed :class:`DeprecationWarning` raised by the usage of `distutils.version.LooseVersion`. :pr:`261` by :user:`Lilian Boulard <LilianBoulard>`

Notes
-----

* Remove trailing imports in the :class:`MinHashEncoder`.

* Fix typos and update links for website.

* Documentation of the :class:`TableVectorizer` and the :class:`SimilarityEncoder` improved.

Dirty-cat Release 0.2.0
=========================

Also see pre-release 0.2.0a1 below for additional changes.

Major changes
-------------

* Bump minimum dependencies:

  - scikit-learn (>=0.21.0) :pr:`202` by :user:`Lilian Boulard <LilianBoulard>`
  - pandas (>=1.1.5) **! NEW REQUIREMENT !** :pr:`155` by :user:`Lilian Boulard <LilianBoulard>`

* **datasets.fetching** - backward-incompatible changes to the example
  datasets fetchers:

  - The backend has changed: we now exclusively fetch the datasets from OpenML.
    End users should not see any difference regarding this.
  - The frontend, however, changed a little: the fetching functions stay the same
    but their return values were modified in favor of a more Pythonic interface.
    Refer to the docstrings of functions `dirty_cat.datasets.fetch_*`
    for more information.
  - The example notebooks were updated to reflect these changes. :pr:`155` by :user:`Lilian Boulard <LilianBoulard>`

* **Backward incompatible change to** :class:`MinHashEncoder`: The :class:`MinHashEncoder` now
  only supports two dimensional inputs of shape (N_samples, 1).
  :pr:`185` by :user:`Lilian Boulard <LilianBoulard>` and :user:`Alexis Cvetkov <alexis-cvetkov>`.

* Update `handle_missing` parameters:

  - :class:`GapEncoder`: the default value "zero_impute" becomes "empty_impute" (see doc).
  - :class:`MinHashEncoder`: the default value "" becomes "zero_impute" (see doc).

  :pr:`210` by :user:`Alexis Cvetkov <alexis-cvetkov>`.

* Add a method "get_feature_names_out" for the :class:`GapEncoder` and the :class:`TableVectorizer`,
  since `get_feature_names` will be depreciated in scikit-learn 1.2. :pr:`216` by :user:`Alexis Cvetkov <alexis-cvetkov>`

Notes
-----

* Removed hard-coded CSV file `dirty_cat/data/FiveThirtyEight_Midwest_Survey.csv`.


* Improvements to the :class:`TableVectorizer`

  - Missing values are not systematically imputed anymore
  - Type casting and per-column imputation are now learnt during fitting
  - Several bugfixes

  :pr:`201` by :user:`Lilian Boulard <LilianBoulard>`

Dirty-cat Release 0.2.0a1
============================

Version 0.2.0a1 is a pre-release.
To try it, you have to install it manually using::

    pip install --pre dirty_cat==0.2.0a1

or from the GitHub repository::

    pip install git+https://github.com/dirty-cat/dirty_cat.git

Major changes
-------------

* Bump minimum dependencies:

  - Python (>= 3.6)
  - NumPy (>= 1.16)
  - SciPy (>= 1.2)
  - scikit-learn (>= 0.20.0)

* :class:`TableVectorizer`: Added automatic transform through the
  :class:`TableVectorizer` class. It transforms
  columns automatically based on their type. It provides a replacement
  for scikit-learn's :class:`~sklearn.compose.ColumnTransformer` simpler to use on heterogeneous
  pandas DataFrame. :pr:`167` by :user:`Lilian Boulard <LilianBoulard>`

* **Backward incompatible change to** :class:`GapEncoder`: The :class:`GapEncoder` now only
  supports two-dimensional inputs of shape (n_samples, n_features).
  Internally, features are encoded by independent :class:`GapEncoder` models,
  and are then concatenated into a single matrix.
  :pr:`185` by :user:`Lilian Boulard <LilianBoulard>` and :user:`Alexis Cvetkov <alexis-cvetkov>`.


Bug-fixes
---------

* Fix `get_feature_names` for scikit-learn > 0.21. :pr:`216` by :user:`Alexis Cvetkov <alexis-cvetkov>`


Dirty-cat Release 0.1.1
========================

Major changes
-------------

Bug-fixes
---------

* RuntimeWarnings due to overflow in :class:`GapEncoder`. :pr:`161` by :user:`Alexis Cvetkov <alexis-cvetkov>`


Dirty-cat Release 0.1.0
=========================

Major changes
-------------

* :class:`GapEncoder`: Added online Gamma-Poisson factorization through the
  :class:`GapEncoder` class. This method discovers latent categories formed
  via combinations of substrings, and encodes string data as combinations of
  these categories. To be used if interpretability is important. :pr:`153` by :user:`Alexis Cvetkov <alexis-cvetkov>`

Bug-fixes
---------

* Multiprocessing exception in notebook. :pr:`154` by :user:`Lilian Boulard <LilianBoulard>`


Dirty-cat Release 0.0.7
========================

* **MinHashEncoder**: Added ``minhash_encoder.py`` and ``fast_hast.py`` files
  that implement minhash encoding through the :class:`MinHashEncoder` class.
  This method allows for fast and scalable encoding of string categorical
  variables.

* **datasets.fetch_employee_salaries**: change the origin of download for employee_salaries.

  - The function now return a bunch with a dataframe under the field "data",
    and not the path to the csv file.
  - The field "description" has been renamed to "DESCR".

* **SimilarityEncoder**: Fixed a bug when using the Jaro-Winkler distance as a
  similarity metric. Our implementation now accurately reproduces the behaviour
  of the ``python-Levenshtein`` implementation.

* **SimilarityEncoder**: Added a `handle_missing` attribute to allow encoding
  with missing values.

* **TargetEncoder**: Added a `handle_missing` attribute to allow encoding
  with missing values.

* **MinHashEncoder**: Added a `handle_missing` attribute to allow encoding
  with missing values.

Dirty-cat Release 0.0.6
=========================

* **SimilarityEncoder**: Accelerate ``SimilarityEncoder.transform``, by:

  - computing the vocabulary count vectors in ``fit`` instead of ``transform``
  - computing the similarities in parallel using ``joblib``. This option can be
    turned on/off via the ``n_jobs`` attribute of the :class:`SimilarityEncoder`.

* **SimilarityEncoder**: Fix a bug that was preventing a :class:`SimilarityEncoder`
  to be created when ``categories`` was a list.

* **SimilarityEncoder**: Set the dtype passed to the ngram similarity
  to float32, which reduces memory consumption during encoding.

Dirty-cat Release 0.0.5
========================

* **SimilarityEncoder**: Change the default ngram range to (2, 4) which
  performs better empirically.

* **SimilarityEncoder**: Added a `most_frequent` strategy to define
  prototype categories for large-scale learning.

* **SimilarityEncoder**: Added a `k-means` strategy to define prototype
  categories for large-scale learning.

* **SimilarityEncoder**: Added the possibility to use hashing ngrams for
  stateless fitting with the ngram similarity.

* **SimilarityEncoder**: Performance improvements in the ngram similarity.

* **SimilarityEncoder**: Expose a `get_feature_names` method.
