import traceback
from unittest.mock import Mock

import pytest
from sklearn.preprocessing import FunctionTransformer

import skrub

# Those tests have to be outside of the `_data_ops` module (and thus out of
# _data_ops/tests/, as we place the tests inside the package), because to
# avoid clutter in the printed stack trace all lines that are inside of the
# _data_ops module are removed: we want the trace of the user's line of code
# where they create the expression, not what happens inside of skrub when they
# do. So if this test was in _data_ops/tests/ the stack description would be
# empty.


def test_creation_stack_description(monkeypatch):
    a = skrub.var("a")
    a = a + 1
    a = a + 1
    a = a + 1  # 'a' was created here
    _ = a + 1
    assert "'a' was created here" in a._skrub_impl.creation_stack_description()
    assert "'a' was created here" in a._skrub_impl.creation_stack_last_line()
    # frames that belong to the boilerplate of the python shell, ipython,
    # jupyter and pytest are also removed.
    assert "pytest" not in a._skrub_impl.creation_stack_description()
    monkeypatch.setattr(
        traceback, "extract_stack", Mock(side_effect=Exception("error"))
    )
    a = skrub.var("a")
    assert a._skrub_impl.creation_stack_description() == ""
    assert a._skrub_impl.creation_stack_last_line() == ""


@pytest.fixture(params=[False, True])
def eval_data_op(request):
    """Fixture to try evaluation both with .skb.eval() and through the learner."""
    if request.param:

        def eval_data_op(data_op, env):
            return data_op.skb.make_learner().fit_transform(env)

    else:

        def eval_data_op(data_op, env):
            return data_op.skb.eval(env)

    return eval_data_op


# Check error message shows where the data_op was defined
def test_apply_eval_failure(eval_data_op):
    a = skrub.var("a")
    b = skrub.var("b")
    c = a / b
    d = c.skb.apply(FunctionTransformer(lambda x: x * 10))
    e = d.skb.apply(FunctionTransformer(lambda x: x + 0.5))
    assert eval_data_op(e, {"a": 1.0, "b": 2.0}) == 5.5
    # error outside of the Apply
    with pytest.raises(
        (ZeroDivisionError, RuntimeError),
        match=(
            r"(?ms).*This node was defined here:.*"
            r"^.*test_data_ops_stack_description\.py.*^.*c = a / b"
        ),
    ):
        eval_data_op(e, {"a": 1.0, "b": 0.0})

    # error in the Apply
    d = c.skb.apply(FunctionTransformer(lambda x: x + "string"))
    e = d.skb.apply(FunctionTransformer(lambda x: x + "something else"))
    with pytest.raises(
        (TypeError, RuntimeError),
        match=r"(?ms).*This node was defined here:.*"
        r"^.*test_data_ops_stack_description\.py.*"
        r'^.*d = c\.skb\.apply\(FunctionTransformer\(lambda x: x \+ "string"\)\)',
    ):
        eval_data_op(e, {"a": 1.0, "b": 2.0})
