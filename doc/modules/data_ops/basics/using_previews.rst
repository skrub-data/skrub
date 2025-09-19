.. currentmodule:: skrub
.. _user_guide_data_ops_using_previews:

Using previews for easier development and debugging
===================================================

To make interactive development easier without having to call ``eval()`` after
each step, it is possible to preview the result of a DataOp by passing a value
along with its name when creating a variable.

>>> import skrub
>>> a = skrub.var("a", 10) # we pass the value 10 in addition to the name
>>> b = skrub.var("b", 6)
>>> c = a + b
>>> c  # now the display of c includes a preview of the result
<BinOp: add>
Result:
―――――――
16

Previews are eager computations on the current data, and since they are computed
immediately they can spot errors early on:

>>> import pandas as pd
>>> df = pd.DataFrame({"col": [1, 2, 3]})
>>> a = skrub.var("a", df)  # we pass the DataFrame as a value

Next, we use the pandas ``drop`` column and try to drop a column without
specifying the axis:

>>> a.drop("col") # doctest: +IGNORE_EXCEPTION_DETAIL +ELLIPSIS
Traceback (most recent call last):
    ...
RuntimeError: Evaluation of '.drop()' failed.
You can see the full traceback above. The error message was:
KeyError: "['col'] not found in axis"

Note that seeing results for the values we provided does *not* change the fact
that we are building a pipeline that we want to reuse, not just computing the
result for a fixed input. The displayed result is only preview of the output on
one example dataset.

>>> c.skb.eval({"a": 3, "b": 2})
5

It is not necessary to provide a value for every variable: it is however advisable
to do so when possible, as it allows to catch errors early on.
