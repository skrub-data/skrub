import pytest

from skrub._drop_similar import DropSimilar


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
        }
    )


@pytest.mark.parametrize(
    "threshold, result",
    [
        (0.8, ["letters", "ranks", "words"]),
        (0.5, ["letters"]),
        (0, []),
    ],
)
def test_drop_similar(df_module, table_with_associations, threshold, result):
    ds = DropSimilar(threshold=threshold)
    res = ds.fit_transform(table_with_associations)
    resulting_columns = res.columns()
    assert resulting_columns == result


def test_wrong_threshold(df_module, table_with_associations):
    ds = DropSimilar(threshold=-0.5)
    with pytest.raises(ValueError):
        ds.fit_transform(table_with_associations)
