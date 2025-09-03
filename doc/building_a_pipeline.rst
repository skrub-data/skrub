.. _user_guide_building_pipeline_index:

.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`


Building a robust machine-learning pipeline
===========================================

Skrub provides two objects that combine the transformers described up to this point
to execute reliable and powerful feature engineering (with the |TableVectorizer|),
and to build a full machine-learning pipeline with good defaults that can be
used as a robust pipeline for most use cases (|tabular_pipeline|).

This section describes the usage of both objects.

.. toctree::
   :maxdepth: 3

   modules/build_pipeline/table_vectorizer
   modules/build_pipeline/tabular_pipeline
