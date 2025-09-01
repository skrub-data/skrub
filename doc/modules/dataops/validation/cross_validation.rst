.. _cross_validation:

Improving the confidence in our score through cross-validation
==============================================================

We can increase our confidence in our score by using cross-validation instead of
a single split. The same mechanism is used but we now fit and evaluate the model
on several splits. This is done with :meth:`.skb.cross_validate()
<DataOp.skb.cross_validate>`.
