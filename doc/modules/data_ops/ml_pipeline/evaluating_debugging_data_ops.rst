.. currentmodule:: skrub
.. _user_guide_data_ops_evaluating_debugging_dataops:

Evaluating and debugging the DataOps plan with :meth:`.skb.full_report() <DataOp.skb.full_report>`
==================================================================================================

All operations on DataOps are recorded in a computational graph, which can be
inspected with :meth:`.skb.full_report() <DataOp.skb.full_report>`. This method
generates a html report that shows the full plan, including all nodes,
their names, descriptions, and the transformations applied to the data.

An example of the report can be found
`here <../../../_static/credit_fraud_report/index.html>`_.

For each node in the plan, the report shows:

- The name and the description of the node, if present.
- Predecessor and successor nodes in the computational graph.
- Where the code relative to the node is defined.
- The estimator fitted in the node along with its parameters (if applicable).
- The preview of the data at that node.

Additionally, if computations fail in the plan, the report shows the offending
node and the error message, which can help in debugging the plan.

By default, reports are saved in the ``skrub_data/execution_reports`` directory, but
they can be saved to a different location with the ``ouptut_dir`` parameter.
Note that the default path can be altered with the
``SKRUB_DATA_DIR`` environment variable. See :ref:`user_guide_configuration_parameters`
for more details.
