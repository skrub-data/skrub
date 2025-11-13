.. |deduplicate| replace:: :func:`~skrub.deduplicate`

.. _user_guide_deduplicate:

Deduplicating categorical data with |deduplicate|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a series or list that contains strings with typos, the |deduplicate|
function may be used to remove the typos. This is done by creating a mapping
between the typo strings and the correct strings.

Deduplication is done by first computing the n-gram distance between unique
categories in data, then performing hierarchical clustering on this distance
matrix, and choosing the most frequent element in each cluster as the
'correct' spelling. This method works best if the true number of
categories is significantly smaller than the number of observed spellings.

>>> from skrub.datasets import make_deduplication_data
>>> duplicated = make_deduplication_data(examples=['black', 'white'],
...                                      entries_per_example=[5, 5],
...                                      prob_mistake_per_letter=0.3,
...                                      random_state=42)
>>> duplicated # doctest: +SKIP
['blacs', 'black', 'black', 'black', 'black', \
'uhibe', 'white', 'white', 'white', 'white']

To deduplicate the data, we can build a correspondence matrix:

>>> from skrub import deduplicate
>>> deduplicate_correspondence = deduplicate(duplicated)
>>> deduplicate_correspondence
blacs    black
black    black
black    black
black    black
black    black
uhibe    white
white    white
white    white
white    white
white    white
dtype: object

>>> deduplicated = list(deduplicate_correspondence)
>>> deduplicated # doctest: +SKIP
['black', 'black', 'black', 'black', 'black', \
'white', 'white', 'white', 'white', 'white']

See the |deduplicate| documentation for caveats and more detail.

Deduplicating values in a dataframe
-----------------------------------

|deduplicate| can be used to replace values in a dataframe that contains typos.
This can be done with ``deduplicate_correspondence`` computed above and the
``map`` function in pandas, or the ``replace`` function in polars.
>>> import pandas as pd
>>> df = pd.DataFrame({'color': duplicated, 'value': range(10)})
>>> df
color  value
0  blacs      0
1  black      1
2  black      2
3  black      3
4  black      4
5  uhibe      5
6  white      6
7  white      7
8  white      8
9  white      9
>>> df['deduplicated_color'] = df['color'].map(deduplicate_correspondence.to_dict())
>>> df
color  value deduplicated_color
0  blacs      0              black
1  black      1              black
2  black      2              black
3  black      3              black
4  black      4              black
5  uhibe      5              white
6  white      6              white
7  white      7              white
8  white      8              white
9  white      9              white

With polars:
>>> import polars as pl  # doctest: +SKIP
>>> df = pl.DataFrame({'color': duplicated, 'value': range(10)}) # doctest: +SKIP
>>> df.with_columns(deduplicated_color = pl.col("color").replace( # doctest: +SKIP
...     deduplicate_correspondence.to_dict())
... )
shape: (10, 3)
┌───────┬───────┬────────────────────┐
│ color ┆ value ┆ deduplicated_color │
│ ---   ┆ ---   ┆ ---                │
│ str   ┆ i64   ┆ str                │
╞═══════╪═══════╪════════════════════╡
│ blacs ┆ 0     ┆ black              │
│ black ┆ 1     ┆ black              │
│ black ┆ 2     ┆ black              │
│ black ┆ 3     ┆ black              │
│ black ┆ 4     ┆ black              │
│ uhibe ┆ 5     ┆ white              │
│ white ┆ 6     ┆ white              │
│ white ┆ 7     ┆ white              │
│ white ┆ 8     ┆ white              │
│ white ┆ 9     ┆ white              │
└───────┴───────┴────────────────────┘
