Getting started
===============

The purpose of this guide is to provide an introduction to the functionalities of ``skrub``, an
open-source package that aims at bridging the gap between tabular data sources and machine-learning models.
Please refer to the `installation guidelines <https://skrub-data.org/stable/install.html>`_ to install ``skrub``.

Much of ``skrub`` revolves around cleaning, assembling, and encoding tabular data, to present data in a format that
machine learning models understand. The following sections will present

Vectorizing data
----------------

``Skrub`` provides tools that can wrangle a complex dataframe in one line of code:

the :class:`~skrub.TableVectorizer`

See end_to_end_pipeline

Assembling data
---------------

``Skrub`` allows imperfect assembly of data, in the case where

+ Example with :class:`~skrub.Joiner`

See the dedicated page on `assembling <https://skrub-data.org/stable/assembling>`_.

Encoding data
-------------

When a column contains dirty categories, it can be encoded using the :class:`~skrub.GapEncoder`.

>>> X = pd.Series(["Rome, Italy", "Rome", "Roma, Italia", "Madrid, SP",
...                "Madrid, spain", "Madrid", "Romq", "Rome, It"], name='city')
>>> enc.fit(X)
GapEncoder(n_components=2, random_state=0)

GapEncoder has found the following two topics:

>>> enc.get_feature_names_out()
['city: madrid, spain, sp', 'city: italia, italy, romq']


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

It could then be used to encode the input column, for instance with a 0 if the city is Madrid, and 1 if the city is Rome:



See the dedicated page on `encoding <https://skrub-data.org/stable/encoding>`_.

Backend interoperability
------------------------

``Skrub`` is designed to work on pandas and polars dataframes, and will be compatible with more backends in the future !
This means that our estimators and tranformers are compatible with both and should have the same behavior, regardless of the module !



Many more
---------

Please refer to our `examples <https://skrub-data.org/stable/auto_examples>`_ for a more in-depth presentation of
the functionalities offered by ``skrub`` !
