.. _subsampling_data:

Subsampling data for easier development and debugging
=====================================================

If the data used for the preview is large, it can be useful to work on a
subsample of the data to speed up the development and debugging process.
This can be done by calling the :meth:`.skb.subsample() <DataOp.skb.subsample>` method
on a variable: this signals to Skrub that what is shown when printing DataOps, or
returned by :meth:`.skb.preview() <DataOp.skb.preview>` is computed on a subsample
of the data.
