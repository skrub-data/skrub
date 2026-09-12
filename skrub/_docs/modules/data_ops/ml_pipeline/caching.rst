.. currentmodule:: skrub
.. _user_guide_data_ops_caching:

Caching for faster recomputation
================================

The results of estimators added to a DataOp with :meth:`.skb.apply()
<DataOp.skb.apply>` and of functions added with :func:`deferred()` or
:meth:`.skb.apply_func() <DataOp.skb.apply_func>` can be cached.

This can save a lot of computation when we run the same operation again. This
typically happens when some step in a pipeline has changed (during
hyperparameter search or because we modified the code), but some earlier steps
remain the same and their results can be reused. For example suppose we have
evaluated the following DataOp:

.. code:: python

          skrub.X().skb.apply(skrub.TableVectorizer()).skb.apply(
              RandomForestRegressor(), y=skrub.y()
          )

and later we modify it to replace the final estimator and run it on the same
data:

.. code:: python

          skrub.X().skb.apply(skrub.TableVectorizer()).skb.apply(
              HistGradientBoostingRegressor(), y=skrub.y()
          )

The transformations applied by the TableVectorizer() can be reused.

For this to happen, we need to enable caching in the configuration, either
``skrub.set_config(cache=True)`` to store cached results in a default location
(a ``_cache`` subdirectory inside the data dir ``get_config()["data_dir"]``)
or ``skrub.set_config(cache="/path/to/cache_dir/")`` to specify where to store
the cache.

.. note::

   The caching mechanism discussed here is about persisting results on disk
   across different evaluations of a DataOp, or evaluations of different
   DataOps. Retaining intermediate results that are used in several places in a
   single DataOp in-memory until they are no longer needed, during a single
   evaluation of the DataOp, always happens.

Forbidding caching for specific nodes
-------------------------------------

When adding nodes to a DataOp, we can specify that their results should never be
cached, even when caching is enabled in the configuration. This is useful if
caching causes errors (e.g. because the arguments or result cannot be
serialized), if result is not deterministic and should be recomputed every time
(for example fetching some information from the network), or if we know that the
function is very fast and caching hinders performance instead of improving it.
This is achieved by passing ``no_cache=True`` to :func:`deferred`,
:meth:`.skb.apply() <DataOp.skb.apply>` or :meth:`.skb.apply_func()
<DataOp.skb.apply_func>`.
