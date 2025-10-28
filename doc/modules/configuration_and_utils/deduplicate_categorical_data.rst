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
