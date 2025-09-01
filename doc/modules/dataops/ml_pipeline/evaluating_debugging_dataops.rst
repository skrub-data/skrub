.. _evaluating_debugging_dataops:

Evaluating and debugging the DataOps plan with :meth:`.skb.full_report() <DataOp.skb.full_report>`
===============================================================================================

All operations on DataOps are recorded in a computational graph, which can be
inspected with :meth:`.skb.full_report() <DataOp.skb.full_report>`. This method
generates a html report that shows the full plan, including all nodes,
their names, descriptions, and the transformations applied to the data.
