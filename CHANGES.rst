:orphan:

.. currentmodule:: dirty_cat

Release 0.4.0
==============

Major changes
-------------

* New experimental feature: joining tables using :func:`fuzzy_join` by approximate key matching. Matches are based
  on string similarities and the nearest neighbors matches are found for each category. :pr:`291` by :user:`Jovan Stojanovic <jovan-stojanovic>` and :user:`Leo Grinsztajn <LeoGrin>`
* **datasets.fetching**: contains a new function :func:`fetch_world_bank_indicator` that can be used to download any indicator from the World Bank Open Data platform. It only needs the indicator ID that can be found on the website. :pr:`291` by :user:`Jovan Stojanovic <jovan-stojanovic>`
* Unnecessary API has been made private: everything (files, functions, classes) starting with an underscore shouldn't be imported in your code. :pr:`331` by :user:`Lilian Boulard <LilianBoulard>`
* The MinHashEncoder now supports a `n_jobs` parameter to parallelize the hashes computation. :pr:`267` by `Leo Grinsztajn <LeoGrin>` and `Lilian Boulard <LilianBoulard>`.

Minor changes
-------------
* Removed example `Fitting scalable, non-linear models on data with dirty categories`. :pr:`386` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* :class:`MinHashEncoder`'s `minhash` method is no longer public. :pr:`379` by :user:`Jovan Stojanovic <jovan-stojanovic>`

Bug fixes
---------

* :class:`MinHashEncoder` now considers `None` and empty strings as missing values, rather
  than raising an error. :pr:`378` by :user:`Gael Varoquaux <GaelVaroquaux>`

Release 0.3.0
=============

Major changes
-------------

* New encoder: :class:`DatetimeEncoder` can transform a datetime column into several numerical columns (year, month, day, hour, minute, second, ...). It is now the default transformer used in the :class:`SuperVectorizer` for datetime columns. :pr:`239` by :user:`Leo Grinsztajn <LeoGrin>`

* The :class:`SuperVectorizer` has seen some major improvements and bug fixes:

  - Fixes the automatic casting logic in ``transform``.
  - To avoid dimensionality explosion when a feature has two unique values, the default encoder (:class:`~sklearn.preprocessing.OneHotEncoder`) now drops one of the two vectors (see parameter `drop="if_binary"`).
  - ``fit_transform`` and ``transform`` can now return unencoded features, like the :class:`~sklearn.compose.ColumnTransformer`'s behavior. Previously, a ``RuntimeError`` was raised.

  :pr:`300` by :user:`Lilian Boulard <LilianBoulard>`

* **Backward-incompatible change in the SuperVectorizer**:
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
    The :class:`SimilarityEncoder` now exclusively uses ``ngram`` for similarities,
    and the `similarity` parameter is deprecated. It will be removed in 0.5. :pr:`282` by :user:`Lilian Boulard <LilianBoulard>`

Notes
-----

* The ``transformers_`` attribute of the SuperVectorizer now contains column
  names instead of column indices for the "remainder" columns. :pr:`266` by :user:`Leo Grinsztajn <LeoGrin>`


Release 0.2.2
=============

Bug fixes
---------

* Fixed a bug in the :class:`SuperVectorizer` causing a `FutureWarning`
  when using the `get_feature_names_out` method. :pr:`262` by :user:`Lilian Boulard <LilianBoulard>`


Release 0.2.1
=============

Major changes
-------------

* Improvements to the :class:`SuperVectorizer`
    - Type detection works better: handles dates, numerics columns encoded as strings, or numeric columns containing strings for missing values.

  :pr:`238` by :user:`Leo Grinsztajn <LeoGrin>`

* `get_feature_names` becomes `get_feature_names_out`, following changes in the scikit-learn API. `get_feature_names` is deprecated in scikit-learn > 1.0. :pr:`241` by :user:`Gael Varoquaux <GaelVaroquaux>`

* Improvements to the :class:`MinHashEncoder`
    - It is now possible to fit multiple columns simultaneously with the :class:`MinHashEncoder`. Very useful when using for instance the :func:`~sklearn.compose.make_column_transformer` method, on multiple columns.
  
  :pr:`243` by :user:`Jovan Stojanovic <jovan-stojanovic>`


Bug-fixes
---------

* Fixed a bug that resulted in the :class:`GapEncoder` ignoring the analyzer argument. :pr:`242` by :user:`Jovan Stojanovic <jovan-stojanovic>`

* :class:`GapEncoder`'s `get_feature_names_out` now accepts all iterators, not just lists. :pr:`255` by :user:`Lilian Boulard <LilianBoulard>`

* Fixed `DeprecationWarning` raised by the usage of `distutils.version.LooseVersion`. :pr:`261` by :user:`Lilian Boulard <LilianBoulard>`

Notes
-----

* Remove trailing imports in the :class:`MinHashEncoder`.

* Fix typos and update links for website.

* Documentation of the SuperVectorizer and the :class:`SimilarityEncoder` improved.

Release 0.2.0
=============

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

* Add a method "get_feature_names_out" for the :class:`GapEncoder` and the :class:`SuperVectorizer`,
  since `get_feature_names` will be depreciated in scikit-learn 1.2. :pr:`216` by :user:`Alexis Cvetkov <alexis-cvetkov>`

Notes
-----

* Removed hard-coded CSV file `dirty_cat/data/FiveThirtyEight_Midwest_Survey.csv`.


* Improvements to the :class:`SuperVectorizer`

  - Missing values are not systematically imputed anymore
  - Type casting and per-column imputation are now learnt during fitting
  - Several bugfixes

  :pr:`201` by :user:`Lilian Boulard <LilianBoulard>`

Release 0.2.0a1
===============

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

* :class:`SuperVectorizer`: Added automatic transform through the
  :class:`SuperVectorizer` class. It transforms
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


Release 0.1.1
=============

Major changes
-------------

Bug-fixes
---------

* RuntimeWarnings due to overflow in :class:`GapEncoder`. :pr:`161` by :user:`Alexis Cvetkov <alexis-cvetkov>`


Release 0.1.0
=============

Major changes
-------------

* :class:`GapEncoder`: Added online Gamma-Poisson factorization through the
  :class:`GapEncoder` class. This method discovers latent categories formed
  via combinations of substrings, and encodes string data as combinations of
  these categories. To be used if interpretability is important. :pr:`153` by :user:`Alexis Cvetkov <alexis-cvetkov>`

Bug-fixes
---------

* Multiprocessing exception in notebook. :pr:`154` by :user:`Lilian Boulard <LilianBoulard>`


Release 0.0.7
=============

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

Release 0.0.6
=============

* **SimilarityEncoder**: Accelerate ``SimilarityEncoder.transform``, by:

  - computing the vocabulary count vectors in ``fit`` instead of ``transform``
  - computing the similarities in parallel using ``joblib``. This option can be
    turned on/off via the ``n_jobs`` attribute of the :class:`SimilarityEncoder`.

* **SimilarityEncoder**: Fix a bug that was preventing a :class:`SimilarityEncoder`
  to be created when ``categories`` was a list.

* **SimilarityEncoder**: Set the dtype passed to the ngram similarity
  to float32, which reduces memory consumption during encoding.

Release 0.0.5
=============

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
