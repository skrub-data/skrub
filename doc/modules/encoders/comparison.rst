.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`
.. |GapEncoder| replace:: :class:`~skrub.GapEncoder`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`

Comparison of the Categorical Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 15 15 25 20 25

    * - Encoder
      - Training time
      - Performance on categorical data
      - Performance on text data
      - Notes
    * - StringEncoder
      - Fast
      - Good
      - Good
      -
    * - TextEncoder
      - Very slow
      - Mediocre to good
      - Very good
      - Requires the ``transformers`` dep.
    * - GapEncoder
      - Slow
      - Good
      - Mediocre to good
      - Interpretable
    * - MinHashEncoder
      - Very fast
      - Mediocre to good
      - Mediocre
      -

:ref:`This example <example_string_encoders>` and this `blog post <https://skrub-data.org/skrub-materials/pages/notebooks/categorical-encoders/categorical-encoders.html>`_ include a more systematic analysis of each method.
