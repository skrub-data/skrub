.. _selectors_ref:

.. currentmodule:: skrub

Selecting columns in a DataFrame
================================

The srkub selectors provide a flexible way to specify the columns on which a
transformation should be applied. They are meant to be used for the ``cols``
argument of :meth:`Expr.skb.apply`, :meth:`Expr.skb.select`,
:meth:`Expr.skb.drop`, :class:`SelectCols` or :class:`DropCols`.

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   selectors.all
   selectors.any_date
   selectors.boolean
   selectors.cardinality_below
   selectors.categorical
   selectors.cols
   selectors.Filter
   selectors.filter
   selectors.filter_names
   selectors.float
   selectors.glob
   selectors.has_nulls
   selectors.integer
   selectors.inv
   selectors.make_selector
   selectors.NameFilter
   selectors.numeric
   selectors.regex
   selectors.select
   selectors.Selector
   selectors.string
