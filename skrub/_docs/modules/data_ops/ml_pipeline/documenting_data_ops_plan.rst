.. currentmodule:: skrub
.. _user_guide_documenting_data_ops_plan:

Documenting the DataOps plan with node names and descriptions
=============================================================

We can improve the readability of the DataOps plan by giving names and descriptions
to the nodes in the plan. This is done with :meth:`.skb.set_name() <DataOp.skb.set_name>`
and :meth:`.skb.set_description() <DataOp.skb.set_description>`.

>>> import skrub
>>> a = skrub.var('a', 1)
>>> b = skrub.var('b', 2)
>>> c = (a + b).skb.set_description('the addition of a and b')
>>> c.skb.description
'the addition of a and b'
>>> d = c.skb.set_name('d')
>>> d.skb.name
'd'

Both names and descriptions can be used to mark relevant parts of the learner, and
they can be accessed from the computational graph and the plan report.

Additionally, names can be used to bypass the computation of a node and override its
result by passing it as a key in the ``environment`` dictionary.

>>> e = d * 10
>>> e
<BinOp: mul>
Result:
―――――――
30
>>> e.skb.eval()
30
>>> e.skb.eval({'a': 10, 'b': 5})
150
>>> e.skb.eval({'d': -1}) # -1 * 10
-10

More info can be found in section :ref:`user_guide_data_ops_truncating_dataplan`.
