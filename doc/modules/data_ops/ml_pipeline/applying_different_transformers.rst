.. currentmodule:: skrub
.. _user_guide_data_ops_applying_different_transformers:

Applying different transformers using skrub selectors and DataOps
=================================================================

It is possible to use skrub selectors to define which columns to apply
transformers to, and then apply different transformers to different subsets of
the data.

For example, this can be useful to apply :class:`~skrub.TextEncoder` to columns
that contain free-flowing text, and :class:`~skrub.StringEncoder` to other string
columns that contain categorical data such as country names.

Or, a string column may need to be encoded in an ordered way, like in the following
example with grades.

>>> import skrub
>>> import pandas as pd
>>> data = {
...     "subject": ["Math", "English", "History", "Science", "Art"],
...     "grade": ["A", "B", "C", "A", "B"]
... }
>>> df = pd.DataFrame(data)
>>> grades = skrub.var("grades", df)
>>> grades
<Var 'grades'>
Result:
―――――――
   subject grade
0     Math     A
1  English     B
2  History     C
3  Science     A
4      Art     B

We encode the subjects with the :class:`~skrub.StringEncoder`:

>>> from skrub import StringEncoder
>>> enc_subject = grades.skb.select(cols="subject").skb.apply(StringEncoder(n_components=2))

For the grades, we define a :func:`~skrub.deferred` function that maps the strings
to the order we want.
Remember that objects inside deferred functions are regular Python
objects (more detail in :ref:`user_guide_data_ops_control_flow`).

>>> @skrub.deferred
... def encode_ordered(df):
...     grade_order = {"A": 3, "B": 2, "C": 1}
...     return df["grade"].map(grade_order)
>>> enc_grades = grades.skb.apply_func(encode_ordered)
>>> enc_grades
<Call 'encode_ordered'>
Result:
―――――――
0    3
1    2
2    1
3    3
4    2
Name: grade, dtype: int64

Finally, we combine the resulting dataframe and series using another deferred
function.

>>> @skrub.deferred
... def combine(subjects, grades):
...     subjects["grade"] = grades
...     return subjects
>>> combine(enc_subject, enc_grades) # doctest: +SKIP
<Call 'combine'>
Result:
―――――――
      subject_0     subject_1  grade
0  1.800470e-07  1.704487e+00      3
1  1.675736e-07 -1.998386e-08      2
2  1.615310e+00  2.142048e-07      1
3 -4.709333e-08  5.155605e-08      3
4 -5.441046e-01  4.167525e-09      2


In the next example, we apply a :class:`~skrub.StringEncoder` to columns
with high cardinality, a mathematical operation to columns with nulls, and a
:class:`~skrub.TableVectorizer` to all other columns. We use the skrub
:ref:`selectors <selectors_ref>` to select the columns based on our requirements.

>>> import pandas as pd
>>> import skrub
>>> orders_df = pd.DataFrame(
...     {
...         "item": ["pen", "cup", "pen", "fork"],
...         "price": [1.5, None, 1.5, 2.2],
...         "qty": [1, 1, 2, 4],
...     }
... )
>>> orders = skrub.var("orders", orders_df)
>>> orders
<Var 'orders'>
Result:
―――――――
   item  price  qty
0   pen    1.5    1
1   cup    NaN    1
2   pen    1.5    2
3  fork    2.2    4

We create some selectors with different conditions:

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
:ref:`user_guide_selectors` and example
:ref:`sphx_glr_auto_examples_09_apply_to_cols.py`.
