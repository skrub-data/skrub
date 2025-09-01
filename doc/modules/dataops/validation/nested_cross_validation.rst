.. _nested_cross_validation:

Validating hyperparameter search with nested cross-validation
=============================================================

To avoid overfitting hyperparameters, the best combination must be evaluated on
data that has not been used to select hyperparameters. This can be done with a
single train-test split or with nested cross-validation.
