.. _user_guide_selectors_with_pandas:

How to combine pandas and skrub selectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`Skrub selectors <user_guide_selectors>` are not limited to skrub
transformers: :meth:`expand` turns a selector into the list of matching column names
of a dataframe, which can then be used with any pandas operation. Because the
selector describes a *rule* rather than a fixed list of names, the same code
keeps working when columns are added to the dataframe.

Here, we aggregate every numeric column except the ``id``, for each group.

>>> import pandas as pd
>>> import skrub.selectors as s
>>> df = pd.DataFrame(
...     {
...         "id": [1, 2, 3, 4],
...         "group": ["A", "A", "B", "B"],
...         "str": ["foo", "bar", "baz", "qux"],
...         "value1": [1, 2, 30, 40],
...         "value2": [5, 6, 7, 8],
...     }
... )

Pick the numeric columns to aggregate, except the ``id``:

>>> only_num = s.numeric() - "id"
>>> only_num
(numeric() - cols('id'))
>>> only_num.expand(df)
['value1', 'value2']

Build the aggregation for those columns and pass it to pandas:

>>> aggs = {col: ["mean", "sum"] for col in only_num.expand(df)}
>>> df.groupby("group").agg(aggs)
      value1     value2
        mean sum   mean sum
group
A        1.5   3    5.5  11
B       35.0  70    7.5  15

When new columns are added, the same selector picks up the new numeric column
and ignores the new string column, with no change to the code:

>>> df["value3"] = [100, 200, 300, 400]
>>> df["note"] = ["a", "b", "c", "d"]
>>> aggs = {col: ["mean", "sum"] for col in only_num.expand(df)}
>>> df.groupby("group").agg(aggs)
      value1     value2     value3
        mean sum   mean sum   mean  sum
group
A        1.5   3    5.5  11  150.0  300
B       35.0  70    7.5  15  350.0  700

The expanded list can be used anywhere pandas expects column names, for
example to select the columns themselves:

>>> df[only_num.expand(df)]
   value1  value2  value3
0       1       5     100
1       2       6     200
2      30       7     300
3      40       8     400

See :ref:`user_guide_selectors_expand` for more on :meth:`expand`, and
:ref:`user_guide_selectors` for the full list of selectors and how to combine
them.
