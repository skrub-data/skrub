import pytest

from skrub._drop_similar import DropSimilar

_THRESHOLD_ERROR = "Threshold must be a number between 0 and 1"


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
    assert ds.kept_cols_ == resulting_columns
    assert resulting_columns == result


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
