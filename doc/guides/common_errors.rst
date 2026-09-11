.. currentmodule:: skrub

.. _common_errors:

Common errors and gotchas
=========================

This page lists errors that users commonly encounter when using skrub, explains
why they happen, and shows how to fix them.

.. contents::
   :local:
   :depth: 1

----

.. _error_single_column_transformer_dataframe:

Passing a dataframe to a SingleColumnTransformer
-------------------------------------------------

**Error**

.. code-block:: text

    ValueError: ``ToDatetime.fit_transform`` should be passed a single column,
    not a dataframe. [...]

**Why it happens**

Transformers that inherit from :class:`~skrub.core.SingleColumnTransformer`
(e.g. :class:`~skrub.ToDatetime`, :class:`~skrub.ToFloat`,
:class:`~skrub.GapEncoder`, :class:`~skrub.MinHashEncoder`) operate on a
single column (a pandas or polars Series), not on a whole dataframe. Passing a
dataframe directly raises a ``ValueError``.

.. code-block:: python

    import pandas as pd
    import skrub

    df = pd.DataFrame({"birthday": ["29/01/2024"], "city": ["London"]})

    # Wrong: passing the full dataframe
    skrub.ToDatetime().fit_transform(df)
    # ValueError: ``ToDatetime.fit_transform`` should be passed a single column...

**Fix**

To apply a :class:`~skrub.core.SingleColumnTransformer` to one or more columns
inside a dataframe, wrap it with :class:`~skrub.ApplyToCols`:

.. code-block:: python

    # Apply to all columns
    skrub.ApplyToCols(skrub.ToDatetime()).fit_transform(df)

    # Apply to a specific column
    skrub.ApplyToCols(skrub.ToDatetime(), cols="birthday").fit_transform(df)

Or pass a single column (Series) directly:

.. code-block:: python

    skrub.ToDatetime().fit_transform(df["birthday"])

**See also**: :class:`~skrub.ApplyToCols`

----

.. _error_cross_validate_data_op:

Passing a DataOp to ``skrub.cross_validate`` instead of a learner
------------------------------------------------------------------

**Error**

.. code-block:: text

    ValueError: `cross_validate` function requires either a Learner object or
    a ParamSearch object, got <class 'skrub._data_ops._data_ops.DataOp'>.

Or a more confusing chained error such as:

.. code-block:: text

    RuntimeError: Evaluation of '.data_op' failed.
    AttributeError: 'Series' object has no attribute 'data_op'

**Why it happens**

:func:`~skrub.cross_validate` expects a fitted-able object — either a
:class:`~skrub.SkrubLearner` (created with :meth:`.skb.make_learner()
<DataOp.skb.make_learner>`) or a :class:`~skrub.ParamSearch` (created with
:meth:`.skb.make_param_search() <DataOp.skb.make_param_search>`). Passing the
DataOp itself instead of a learner is a common mistake:

.. code-block:: python

    import skrub

    X = skrub.X()
    y = skrub.y()
    pred = X.skb.apply(some_estimator, y=y)

    # Wrong: passing the DataOp directly
    skrub.cross_validate(pred, pred.skb.get_data())

**Fix**

Call :meth:`.skb.make_learner() <DataOp.skb.make_learner>` on your DataOp
first, then pass the resulting learner to :func:`~skrub.cross_validate`:

.. code-block:: python

    learner = pred.skb.make_learner()
    skrub.cross_validate(learner, pred.skb.get_data())

**See also**: :meth:`DataOp.skb.make_learner`, :func:`~skrub.cross_validate`

----

.. _error_choices_clamped:

Hyperparameter choices being silently clamped
---------------------------------------------

**Warning**

.. code-block:: text

    UserWarning: The following choices are used in the construction of X or y,
    so their value cannot be tuned because they are needed outside of the
    cross-validation loop. They will be clamped to their default value: [...]

**Why it happens**

In a DataOps pipeline, :func:`~skrub.choose_from` and similar choice
constructors define tunable hyperparameters. However, choices that appear
*upstream* of :meth:`.skb.mark_as_X() <DataOp.skb.mark_as_X>` or
:meth:`.skb.mark_as_y() <DataOp.skb.mark_as_y>` are part of the feature
matrix or target construction. Because the features and target must be fixed
before the cross-validation loop begins (to split data consistently), those
choices cannot vary across folds and are therefore frozen to their default
value.

.. code-block:: python

    import skrub

    orders = skrub.var("orders", orders_df)

    # Wrong: the choice is upstream of mark_as_X()
    n_components = skrub.choose_int(5, 50, name="n_components")
    X = orders.skb.apply(skrub.TableVectorizer(high_card_cat_transformer=skrub.GapEncoder(n_components=n_components)))
    X = X.skb.mark_as_X()
    # This choice will be clamped and cannot be tuned.

**Fix**

Move tunable choices to *after* :meth:`.skb.mark_as_X()
<DataOp.skb.mark_as_X>` and :meth:`.skb.mark_as_y()
<DataOp.skb.mark_as_y>`, so they are inside the cross-validation loop:

.. code-block:: python

    X = orders.skb.apply(skrub.TableVectorizer()).skb.mark_as_X()

    # The choice is now downstream of mark_as_X(), so it can be tuned
    n_components = skrub.choose_int(5, 50, name="n_components")
    X = X.skb.apply(skrub.GapEncoder(n_components=n_components), cols="some_col")

If the choice genuinely needs to be upstream (e.g. it controls how raw data is
loaded or filtered), accept that it cannot be cross-validated and set it
manually.

**See also**: :func:`~skrub.choose_from`, :meth:`DataOp.skb.mark_as_X`,
:ref:`user_guide_data_ops_hyperparameter_tuning`

----

.. _error_graphviz_no_formats:

Graphviz is installed but ``draw_data_op_graph`` raises an error
----------------------------------------------------------------

**Error**

.. code-block:: text

    ImportError: Please install pydot and graphviz to draw data_op graphs.

Even though graphviz *is* already installed, and the actual underlying error
from the graphviz executable is:

.. code-block:: text

    Format: "svg" not recognized. No formats found.
    Perhaps "dot -c" needs to be run (with installer's privileges)
    to register the plugins?

**Why it happens**

Graphviz stores its rendering plugins in a cache that must be initialised
after installation. If this step was skipped (e.g. when installing via conda
or pip into a fresh environment), the ``dot`` binary exists but cannot produce
any output format, causing skrub's import check to fail with a misleading
message.

**Fix**

Run the following command **in the same environment** where skrub is installed:

.. code-block:: bash

    dot -c

This registers the graphviz plugins. After running it, ``draw_data_op_graph``
should work without restarting Python.

**See also**: :func:`~skrub.draw_data_op_graph`

----

.. _error_node_evaluation_failed:

Understanding "Evaluation of node ... failed" errors
-----------------------------------------------------

**Error**

.. code-block:: text

    RuntimeError: Evaluation of node <Apply some_function> failed. See above
    for full traceback. This node was defined here:
      File "my_script.py", line 42, in <module>

**Why it happens**

When a DataOp graph is evaluated (at :meth:`~skrub.SkrubLearner.fit` time or
when calling :meth:`.skb.eval() <DataOp.skb.eval>`), skrub catches exceptions
raised inside any graph node and re-raises them with context about *which node*
failed and *where it was defined* in your code. The ``RuntimeError`` is a
wrapper; the original error is shown in the traceback above it.

This wrapping can make the traceback look long and intimidating, but the
relevant information is always:

1. **The original exception** — shown first, above the ``RuntimeError``.
2. **"This node was defined here"** — the line in your code that created the
   failing node.

**Fix**

Read the full traceback from the top. The first exception listed is the real
error (e.g. a ``KeyError``, ``ValueError``, or ``TypeError`` from your own
function). Fix that underlying error; the ``RuntimeError`` wrapper will
disappear automatically.

When debugging inside cross-validation, pass ``error_score="raise"`` to
:func:`~skrub.cross_validate` to prevent scikit-learn from silently swallowing
errors in individual folds:

.. code-block:: python

    skrub.cross_validate(learner, env, error_score="raise")

**See also**: :meth:`DataOp.skb.eval`, :func:`~skrub.cross_validate`

----
