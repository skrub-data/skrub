.. _control_flow_dataops:

Control flow in DataOps: eager and deferred evaluation
======================================================

DataOps represent computations that have not been executed yet, and will
only be triggered when we call :meth:`.skb.eval() <DataOp.skb.eval>`, or when we
create the pipeline with :meth:`.skb.make_learner() <DataOp.skb.make_learner>` and
call one of its methods such as ``fit()``.

This means we cannot use standard Python control flow statements such as ``if``,
``for``, ``with``, etc. with DataOps, because those constructs would execute
immediately.
