.. _can_i_use_library_x:

Can I use library "x" with Skrub DataOps?
==========================================

Yes, Skrub DataOps are designed to be "transparent", so that any method used by
the underlying data structures (e.g., Pandas or Polars) can be accessed directly:
check :ref:`user_guide_direct_access_ref` for more details.
All DataOps-specific operations are available through the ``.skb`` attribute,
which provides access to the DataOps namespace. Other library-specific methods
are available directly from the DataOp object, as if it were a regular object
(like a Pandas or Polars DataFrame or Series).
