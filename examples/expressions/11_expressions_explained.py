"""
Skrub expressions
=================

"""

# %%
# General workflow
# ----------------


# %%
# Declare inputs with ``skrub.var``

# %%
import skrub

a = skrub.var("a")
b = skrub.var("b")

# %%
# Apply transformations, composing more complex expressions

# %%
e = a + b
e

# %%
# We can evaluate the expression. (Note not very often used in practice but we
# use it in the example for explaining how expressions work.)

# %%
e.skb.eval({"a": 10, "b": 6})

# %%
e.skb.eval({"a": 2, "b": 3})

# %%
# Get a scikit-learn estimator that can be fitted and applied to data

# %%
estimator = e.skb.get_estimator()
estimator.fit_transform({"a": 2, "b": 3})

# %%
# Composing expressions
# ---------------------
#
# The simplest expressions are variables created with ``skrub.var`` (or plain
# python objects). Complex expressions are constructed by applying functions or
# operators to other expressions.
#
# Those operations are remembered and applied directly to the data we provide
# to the pipeline. So if we know for example that we have a variable for which
# we will pass a numpy array, we can use it just as we would use a numpy array.
#
# More transformations (eg applying a scikit-learn estimator) are available
# through the special ``skb`` attribute, as we will see later.

# %%
lengths_mm = skrub.var("mm")

# arithmetic operators
lengths_m = lengths_mm / 1000

# indexing, slicing, getting items
lengths_m = lengths_m[::2]

# accessing attributes, calling methods
lengths_m = lengths_m.reshape(-1, 1)
lengths_m

# %%
import numpy as np

lengths_m.skb.eval({"mm": np.array([100.0, 200.0, 300.0, 400.0, 500.0])})

# %%
# What if we want to apply a function to our expression?
# The following raises an error::
#
#     clipped_lengths = np.minimum(lengths_m, 0.3) # noqa
#
# Produces:
#
# ``TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly use its Boolean value now.`` # noqa: E501

# %%
# Indeed ``lengths_m`` is not a numpy array, it is a skrub expression that will
# produce a numpy array when evaluated. So we need to defer the call to
# ``np.minimum`` until we have an actual argument to give it.

# %%
clipped_lengths = skrub.deferred(np.minimum)(lengths_m, 0.3)
clipped_lengths

# %%
clipped_lengths.skb.eval({"mm": np.array([100.0, 200.0, 300.0, 400.0, 500.0])})

# %%
# Delayed evaluation
# ------------------
#
# What we saw above is important to understand: expressions represent a
# computation that has not yet been executed, and will be executed when we
# trigger it, for example by calling ``eval()`` or getting the estimator and
# calling ``fit()``.
#
# Why must the computation be delayed? We are not interested in the result on
# one particular input, we want to build a pipeline that can be fitted and
# applied to many inputs.
#
# This means that we cannot use usual Python control flow statements with
# expressions, because those would execute immediately.

# %%
colors = skrub.var("colors")
colors.skb.eval({"colors": ["red", "green", "blue"]})

# %%
# We cannot do this::
#
#     for col in colors:
#         pass
#
# This would result in:
#
# ``TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly iterate over it now.`` # noqa: E501
#

# %%
# The solution is to use ``skrub.deferred``
#
# bad::
#
#     upper_colors = [c.upper() for c in colors]
#
# This would result in:
#
# ``TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly iterate over it now.`` # noqa: E501

# %%
# good:


# %%
@skrub.deferred
def upper(colors):
    return [c.upper() for c in colors]


colors = upper(colors)
colors

# %%
colors.skb.eval({"colors": ["red", "green", "blue"]})

# %%
# similarly, all steps must not transform the input in-place but return a value

# %%
colors.append("YELLOW")

# %%
colors.skb.eval({"colors": ["red", "green", "blue"]})


# %%
# instead

# %%
colors = colors + ["YELLOW"]

# %%
# or


# %%
@skrub.deferred
def append_pink(colors):
    new_colors = []
    for c in colors:
        new_colors.append(c)
    new_colors.append("PINK")
    return new_colors


colors = append_pink(colors)
colors

# %%
colors.skb.eval({"colors": ["red", "green", "blue"]})


# %%
# Initializing variables with a value
# -----------------------------------
#
# Repeatedly calling ``eval()`` as we have been doing is inconvenient. But we
# can initialize variables with a value. When we do so, when an expression is
# created, it evaluates the result on available data and shows it to us. Any
# errors are triggered and we get better tab-completion.
#
# **But** that does not change the fact that we are building a pipeline that we
# want to reuse, not just computing the result for a fixed input. So all the
# limitations described above still apply. Think of the displayed result as a
# preview of the pipeline's output on one example dataset.

# %%
color = skrub.var("color", value="red")
color

# %%
color = color.upper()
message = "The color is: " + color
message


# %%
message.skb.eval({"color": "green"})

# %%
message.skb.eval({"color": "blue"})

# %%
# Applying scikit-learn estimators
# --------------------------------
#
# - Start example on a real dataset
# - show ``.skb.apply``
# - now we have a pipeline interesting enough to showcase `.skb.full_report()`
#
# Cross-validation
# ----------------
#
# - explain ``.skb.mark_as_x()``, ``.skb.mark_as_y()``, ``skrub.X()``, ``skrub.y()``
# - show ``.skb.cross_validate()``
# - show ``.skb.get_estimator()`` again
#
# Conclusion
# ----------
#
# There is more: see the next example for hyperparameter search.
#
# A few more advanced features have not been shown and remain for more
# specialized examples, for example:
#
# - naming nodes, passing the value for any node with the inputs
# - ``.skb.applied_estimator``
# - ``.skb.concat_horizontal``, ``.skb.drop`` and ``.skb.select``, skrub selectors
# - ``.skb.freeze_after_fit`` (niche / very advanced)
