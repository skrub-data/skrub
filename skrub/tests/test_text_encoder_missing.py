"""Test TextEncoder behavior when dependencies are missing."""
import pandas as pd
import pytest

from skrub import TextEncoder


def test_missing_import_error():
    """Test error message when sentence_transformers is not installed."""
    try:
        import sentence_transformers  # noqa
    except ImportError:
        pass
    else:
        pytest.skip("sentence_transformers is installed, skipping this test")

    encoder = TextEncoder()
    x = pd.Series(["oh no"])

    err_msg = (
        "Missing optional dependency 'sentence_transformers'.*"
        "TextEncoder requires sentence-transformers.*"
        "install\\.html#deep-learning-dependencies"
    )

    with pytest.raises(ImportError, match=err_msg):
        encoder.fit(x)
