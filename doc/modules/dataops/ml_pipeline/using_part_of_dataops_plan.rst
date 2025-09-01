.. _using_part_of_dataops_plan:

Using only a part of a DataOps plan
===================================

Besides documenting a DataOps plan, the :meth:`.skb.set_name() <DataOp.skb.set_name>`
has additional functions. By setting a name, we can:

- Bypass the computation of that node and override its result by passing it as a
  key in the ``environment`` argument.
- Truncate the computational graph after this node to obtain the intermediate result with
  :meth:`SkrubLearner.truncated_after`.
- Retrieve that node and inspect the estimator that was fitted in it, if the
  node was created with :meth:`.skb.apply() <DataOp.skb.apply>`.
