import builtins
import sys

import pytest

from skrub._drop_similar import DropSimilar
from skrub.conftest import skip_polars_installed_without_pyarrow

_THRESHOLD_ERROR = "Threshold must be a number between 0 and 1"
_PYARROW_ERROR = "DropSimilar requires the Pyarrow package to run on Polars dataframes."


@pytest.fixture
def table_with_associations(df_module):
    return df_module.make_dataframe(
        {
            "letters": [
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "a",
            ],
            "ranks": [
                "first",
                "second",
                "third",
                "fourth",
                "second",
                "third",
                "fourth",
                "first",
                "second",
                "first",
            ],
            "words": [
                "None",
                "None",
                "None",
                "Other",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
            ],
            "more_words": [
                "None",
                "None",
                "None",
                "Other",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
            ],
        }
    )


@skip_polars_installed_without_pyarrow
@pytest.mark.parametrize(
    "threshold, result",
    [
        (1.0, ["letters", "ranks", "words"]),
        (0.8, ["letters", "ranks", "words"]),
        (0.5, ["letters"]),
        (0, ["letters"]),
    ],
)
def test_drop_similar(table_with_associations, threshold, result):
    ds = DropSimilar(threshold=threshold)
    res = ds.fit_transform(table_with_associations)
    resulting_columns = list(res.columns)
    kept_cols = list(ds.get_feature_names_out())
    assert kept_cols == resulting_columns
    assert resulting_columns == result


@skip_polars_installed_without_pyarrow
def test_wrong_threshold(df_module):
    ds = DropSimilar(threshold=-0.5)
    with pytest.raises(ValueError, match=_THRESHOLD_ERROR):
        ds.fit_transform(df_module.make_dataframe({}))
    ds = DropSimilar(threshold=3)
    with pytest.raises(ValueError, match=_THRESHOLD_ERROR):
        ds.fit_transform(df_module.make_dataframe({}))
    ds = DropSimilar(threshold=False)
    with pytest.raises(ValueError, match=_THRESHOLD_ERROR):
        ds.fit_transform(df_module.make_dataframe({}))
    ds = DropSimilar(threshold="lower")
    with pytest.raises(ValueError, match=_THRESHOLD_ERROR):
        ds.fit_transform(df_module.make_dataframe({}))


def test_without_pyarrow(monkeypatch):
    pl = pytest.importorskip("polars")
    example_dataframe = pl.DataFrame({"a": [1, 2, 3]})
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    assert "pyarrow" not in sys.modules
    ds = DropSimilar()

    builtin_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError(name)
        return builtin_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ImportError, match=_PYARROW_ERROR):
        ds.fit_transform(example_dataframe)
