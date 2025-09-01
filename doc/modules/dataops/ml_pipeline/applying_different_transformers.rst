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

>>> from skrub import selectors as s
>>> high_cardinality = s.string() - s.cardinality_below(2)
>>> has_nulls = s.has_nulls()
>>> leftover = s.all() - high_cardinality - has_nulls

>>> vectorizer = skrub.StringEncoder(n_components=2)
>>> vectorized_items = orders.skb.select(high_cardinality).skb.apply(vectorizer)
>>> vectorized_items # doctest: +SKIP
<Apply StringEncoder>
Result:
―――――――
          item_0        item_1  price  qty
0  1.511858e+00  9.380015e-08    1.5    1
1 -1.704687e-07  1.511858e+00    NaN    1
2  1.511858e+00  9.380015e-08    1.5    2
3 -5.458670e-09 -6.917769e-08    2.2    4

>>> vectorized_has_nulls = orders.skb.select(cols=has_nulls) * 11
>>> vectorized_has_nulls
    <BinOp: mul>
    Result:
    ―――――――
       price
    0   16.5
    1    NaN
    2   16.5
    3   24.2
>>> everything_else = orders.skb.select(cols=leftover).skb.apply(skrub.TableVectorizer())

After encoding the columns, the resulting DataOps can be concatenated together
to obtain the final result:

>>> encoded = (
...   everything_else.skb.concat([vectorized_items, vectorized_has_nulls], axis=1)
... )
>>> encoded # doctest: +SKIP
   qty        item_0        item_1  price
0  1.0  1.594282e+00 -1.224524e-07   16.5
1  1.0  9.228692e-08  1.473794e+00    NaN
2  2.0  1.594282e+00 -1.224524e-07   16.5
3  4.0  7.643604e-09  6.080018e-01   24.2

More info on advanced column selection and manipulation be found in
:ref:`userguide_selectors` and example
:ref:`sphx_glr_auto_examples_10_apply_on_cols.py`.
