.. _expressions_ref:

.. currentmodule:: skrub

Skrub Expressions
=================

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   as_expr
   choose_bool
   choose_float
   choose_from
   choose_int
   cross_validate
   deferred
   eval_mode
   optional
   train_test_split
   var
   X
   y

.. autosummary::
   :toctree: generated/
   :template: expr_class.rst
   :nosignatures:

   Expr

The ``skb`` accessor is available for all expressions, and provides

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst
   :nosignatures:

   Expr.skb.apply
   Expr.skb.apply_func
   Expr.skb.clone
   Expr.skb.concat
   Expr.skb.cross_validate
   Expr.skb.describe_defaults
   Expr.skb.describe_param_grid
   Expr.skb.describe_steps
   Expr.skb.draw_graph
   Expr.skb.drop
   Expr.skb.eval
   Expr.skb.freeze_after_fit
   Expr.skb.full_report
   Expr.skb.get_data
   Expr.skb.get_pipeline
   Expr.skb.get_grid_search
   Expr.skb.get_randomized_search
   Expr.skb.if_else
   Expr.skb.iter_pipelines_grid
   Expr.skb.iter_pipelines_randomized
   Expr.skb.mark_as_X
   Expr.skb.mark_as_y
   Expr.skb.match
   Expr.skb.preview
   Expr.skb.select
   Expr.skb.set_description
   Expr.skb.set_name
   Expr.skb.subsample
   Expr.skb.train_test_split

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst
   :nosignatures:

   Expr.skb.description
   Expr.skb.is_X
   Expr.skb.is_y
   Expr.skb.name
   Expr.skb.applied_estimator

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   SkrubPipeline
   ParamSearch
