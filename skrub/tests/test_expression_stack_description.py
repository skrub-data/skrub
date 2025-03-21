import traceback
from unittest.mock import Mock

import skrub

# This test has to be outside of the `_expressions` module (and thus out of
# _expressions/tests/, as we place the tests inside the package), because to
# avoid clutter in the printed stack trace all lines that are inside of the
# _expressions module are removed: we want the trace of the user's line of code
# where they create the expression, not what happens inside of skrub when they
# do. So if this test was in _expressions/tests/ the stack description would be
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
