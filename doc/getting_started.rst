Getting started
===============

The purpose of this guide is to provide an introduction to the functionalities of ``skrub``, an
open-source package that aims at bridging the gap between tabular data sources and machine-learning models.
Please refer to our `installation guidelines <https://skrub-data.org/stable/install.html>`_ for installing ``skrub``.

Much of ``skrub`` revolves around vectorizing, assembling, and encoding tabular data, to prepare data in a format that
machine-learning models understand.


Creating a machine-learning pipeline
------------------------------------

:func:`~skrub.tabular_learner`


For more information about the chosen defaults, visit: `end-to-end pipeline <https://skrub-data.org/stable/end_to_end_pipeline>`_.
Feel free to try it out on one of our `datasets <https://skrub-data.org/stable/reference/downloading_a_dataset>`_.

Vectorizing data
----------------

``Skrub`` provides the :class:`~skrub.TableVectorizer`, a tool that wrangle a complex dataframe in one line of code.

``Skrub`` is designed to work on pandas and polars dataframes, and will be compatible with more backends in the future.
We are working towards making sure our estimators are compatible with both and have the same behavior, regardless of the module !



Assembling data
---------------

``Skrub`` allows imperfect assembly of data, in the case where columns are dirty and can contain typos. The :class:`~skrub.Joiner`
allows to fuzzy-join multiple tables, and each row of a main table will be augmented with values from the best match
in the auxiliary table. You can control how distant fuzzy-matches are allowed to be with the ``max_dist`` parameter.

In the following, we add information about capitals to a table of countries:

>>> import pandas as pd
>>> from skrub import Joiner
>>> main_table = pd.DataFrame({"Country": ["France", "Italia", "Georgia"]})
>>> aux_table = pd.DataFrame( {"Country": ["Germany", "France", "Italy"],
...                            "Capital": ["Berlin", "Paris", "Rome"]} )
>>> main_table
  Country
0  France
1  Italia
2   Georgia
>>> aux_table
   Country Capital
0  Germany  Berlin
1   France   Paris
2    Italy    Rome
>>> joiner = Joiner(
...     aux_table,
...     key="Country",  # here both tables have the same key. You can also use left_key and right_key
...     suffix="_aux",
...     max_dist=0.8,
...     add_match_info=False,
... )
>>> joiner.fit_transform(main_table)
  Country      Country_aux      Capital_aux
0  France           France            Paris
1  Italia            Italy             Rome
2  Georgia              NaN              NaN  # not present in the auxiliary table

It's also possible to augment data by joining and aggregating multiple dataframes with the :class:`~skrub.AggJoiner`. This is
particularly useful to summarize information scattered across tables:

>>> import pandas as pd
>>> from skrub import AggJoiner
>>> main = pd.DataFrame({
...     "airportId": [1, 2],
...     "airportName": ["Paris CDG", "NY JFK"],
... })
>>> aux = pd.DataFrame({
...     "flightId": range(1, 7),
...     "from_airport": [1, 1, 1, 2, 2, 2],
...     "total_passengers": [90, 120, 100, 70, 80, 90],
...     "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
... })
>>> agg_joiner = AggJoiner(
...     aux_table=aux,
...     main_key="airportId",
...     aux_key="from_airport",
...     cols=["total_passengers", "company"],
...     operations=["mean", "mode"],  # compute the mode of categorical features and the mean of numerical features
... )
>>> agg_joiner.fit_transform(main)
   airportId  airportName  company_mode  total_passengers_mean
0          1    Paris CDG            AF              103.33...
1          2       NY JFK            DL               80.00...

See other ways to join multiple tables on `assembling data <https://skrub-data.org/stable/assembling>`_.


Encoding data
-------------

When a column contains dirty categories, it can be encoded using one of ``skrub``'s encoders, such as
the :class:`~skrub.GapEncoder`.

The :class:`~skrub.GapEncoder` creates a continuous encoding, based on the activation of latent categories. It
will create the encoding based on combinations of substrings which frequently co-occur.

For instance, we might want to encode a column ``X`` that we know contains information about cities, being
either Madrid or Rome :

>>> X = pd.Series(["Rome, Italy", "Rome", "Roma, Italia", "Madrid, SP",
...                "Madrid, spain", "Madrid", "Romq", "Rome, It"], name="city")
>>> enc.fit(X)
GapEncoder(n_components=2, random_state=0)

The GapEncoder has found the following two topics:

>>> enc.get_feature_names_out()
['city: madrid, spain, sp', 'city: italia, italy, romq']

Which correspond to the two cities.

Let's see the activation of each topic in each of the rows of ``X``:

>>> out = enc.transform(X)
>>> out
   city: madrid, spain, sp  city: italia, italy, romq
0                 0.052257                  13.547743
1                 0.050202                   3.049798
2                 0.063282                  15.036718
3                12.047028                   0.052972
4                16.547818                   0.052182
5                 6.048861                   0.051139
6                 0.050019                   3.049981
7                 0.053193                   9.046807

The higher the activation, the closer the row to the latent topic. These activations can then be used to encode
``X``, for instance with a 0 if the city is Madrid, and 1 if the city is Rome:

>>> madrid = out.iloc[:,0] > out.iloc[:,1]
>>> X[madrid] = 0
>>> X[~madrid] = 1
0    1
1    1
2    1
3    0
4    0
5    0
6    1
7    1
Name: city, dtype: object

Which corresponds to respective positions of Madrid and Rome in the initial column ! This column can now be understood
by a machine-learning model.

The other encoders are presented in `encoding <https://skrub-data.org/stable/encoding>`_.


Next steps
----------

We have briefly covered pipeline creation, vectorizing, assembling, and encoding. We presented the main functionalities of ``skrub``,
but there is much more to it !

Please refer to our `User Guide <https://skrub-data.org/stable/documentation>`_ for a more in-depth presentation of
``skrub``'s concepts. You can also check out our `API reference <https://skrub-data.org/stable/api>`_ for the exhaustive
list of functionalities !

Visit our `examples <https://skrub-data.org/stable/auto_examples>`_ for more illustrations of the tools offered by ``skrub``.
