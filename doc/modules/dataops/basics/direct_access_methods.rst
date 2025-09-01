.. _direct_access_methods:

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

>>> orders = skrub.var("orders", orders_df)

Because we know that a dataframe will be provided as input to the computation, we
can manipulate ``orders`` as if it were a regular dataframe.
