.. _exporting_dataops:

Exporting the DataOps plan as a learner and reusing it
======================================================

DataOps are designed to build complex pipelines that can be reused on new, unseen
data in potentially different environments from where they were created. This can
be achieved by exporting the DataOps plan as a **learner**: the learner is special
transformer that is similar to a scikit-learn estimator, but that takes as input
the **environment** that should be used to execute the operations.
