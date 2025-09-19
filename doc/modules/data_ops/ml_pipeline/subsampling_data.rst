.. currentmodule:: skrub
.. _user_guide_data_ops_subsampling:

Subsampling data for easier development and debugging
=====================================================

If the data used for the preview is large, it can be useful to work on a
subsample of the data to speed up the development and debugging process.
This can be done by calling the :meth:`.skb.subsample() <DataOp.skb.subsample>` method
on a variable: this signals to skrub that what is shown when printing DataOps, or
returned by :meth:`.skb.preview() <DataOp.skb.preview>` is computed on a subsample
of the data.

Note that subsampling is "local": if it is applied to a variable, it only
affects the variable itself. This may lead to unexpected results and errors
if, for example, ``X`` is subsampled but ``y`` is not.

Subsampling **is turned off** by default when we call other methods such as
:meth:`.skb.eval() <DataOp.skb.eval>`,
:meth:`.skb.cross_validate() <DataOp.skb.cross_validate>`,
:meth:`.skb.train_test_split <DataOp.skb.train_test_split>`,
:meth:`DataOp.skb.make_learner`,
:meth:`DataOp.skb.make_randomized_search`, etc.
However, all of those methods have a ``keep_subsampling`` parameter that we can
set to ``True`` to force using the subsampling when we call them. Note that
even if we set ``keep_subsampling=True``, subsampling is not applied when using
``predict``.

See more details in a :ref:`full example <sphx_glr_auto_examples_data_ops_14_subsampling.py>`.
