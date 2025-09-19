.. currentmodule:: skrub
.. _user_guide_direct_access_ref:

DataOps allow direct access to methods of the underlying data
=============================================================

DataOps are designed to be flexible and allow direct access to the underlying data,
so that it is possible to use the APIs of the underlying data structures
(e.g., Pandas or Polars) directly:

Suppose we want to process dataframes that look like this:

>>> import pandas as pd
>>> orders_df = pd.DataFrame(
...     {
...         "item": ["pen", "cup", "pen", "fork"],
...         "price": [1.5, None, 1.5, 2.2],
...         "qty": [1, 1, 2, 4],
...     }
... )
>>> orders_df
   item  price  qty
0   pen    1.5    1
1   cup    NaN    1
2   pen    1.5    2
3  fork    2.2    4

We can create a skrub variable to represent that input:

>>> import skrub
>>> orders = skrub.var("orders", orders_df)

Because we know that a dataframe will be provided as input to the computation, we
can manipulate ``orders`` as if it were a regular dataframe.

We can access its attributes:

>>> orders.columns
<GetAttr 'columns'>
Result:
―――――――
Index(['item', 'price', 'qty'], dtype='object')

Accessing items, indexing, slicing:

>>> orders["item"].iloc[1:]
<GetItem slice(1, None, None)>
Result:
―――――――
1     cup
2     pen
3    fork
Name: item, dtype: object

We can apply operators:

>>> orders["price"] * orders["qty"]
<BinOp: mul>
Result:
―――――――
0    1.5
1    NaN
2    3.0
3    8.8
dtype: float64

We can call methods:

>>> orders.assign(total=orders["price"] * orders["qty"])
<CallMethod 'assign'>
Result:
―――――――
   item  price  qty  total
0   pen    1.5    1    1.5
1   cup    NaN    1    NaN
2   pen    1.5    2    3.0
3  fork    2.2    4    8.8

Note that the original ``orders`` variable is not modified by the operations
above. Instead, each operation creates a new DataOp. DataOps cannot be
modified in-place, all operations that we apply must produce a new value.
