.. _applying_different_transformers:

Applying different transformers using Skrub selectors and DataOps
=================================================================

It is possible to use Skrub selectors to define which columns to apply
transformers to, and then apply different transformers to different subsets of
the data.

For example, this can be useful to apply :class:`~skrub.TextEncoder` to columns
that contain free-flowing text, and :class:`~skrub.StringEncoder` to other string
columns that contain categorical data such as country names.

In the example below, we apply a :class:`~skrub.StringEncoder` to columns
with high cardinality, a mathematical operation to columns with nulls, and a
:class:`~skrub.TableVectorizer` to all other columns.
