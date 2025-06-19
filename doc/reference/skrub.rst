:html_theme.sidebar_secondary.remove:


.. _skrub_ref:


skrub
=====

.. currentmodule:: skrub



.. _skrub_ref-building-a-pipeline:



Building a pipeline
-------------------



For more flexibility and control to build pipelines, see the :ref:`skrub expressions <expressions_ref>`.


.. autosummary::
  :nosignatures:
  :toctree: generated/
  :template: base.rst


  tabular_learner
  TableVectorizer
  Cleaner
  SelectCols
  DropCols
  DropUninformative


.. _skrub_ref-encoding-a-column:



Encoding a column
-----------------



See :ref:`encoding <encoding>` for further details.


.. autosummary::
  :nosignatures:
  :toctree: generated/
  :template: base.rst


  StringEncoder
  TextEncoder
  MinHashEncoder
  GapEncoder
  SimilarityEncoder
  ToCategorical
  DatetimeEncoder
  ToDatetime
  to_datetime


.. _skrub_ref-generating-an-html-report:



Generating an HTML report
-------------------------




.. autosummary::
  :nosignatures:
  :toctree: generated/
  :template: base.rst


  TableReport
  patch_display
  unpatch_display
  column_associations


.. _skrub_ref-cleaning-a-dataframe:



Cleaning a dataframe
--------------------




.. autosummary::
  :nosignatures:
  :toctree: generated/
  :template: base.rst


  deduplicate


.. _skrub_ref-joining-dataframes:



Joining dataframes
------------------




.. autosummary::
  :nosignatures:
  :toctree: generated/
  :template: base.rst


  Joiner
  AggJoiner
  MultiAggJoiner
  AggTarget
  InterpolationJoiner
  fuzzy_join
