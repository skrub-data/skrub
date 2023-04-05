import pandas as pd
import pytest

from dirty_cat._utils import LRUDict, _infer_date_format


def test_lrudict():
    dict_ = LRUDict(10)

    for x in range(15):
        dict_[x] = f"filled {x}"

    for x in range(5, 15):
        assert x in dict_
        assert dict_[x] == f"filled {x}"

    for x in range(5):
        assert x not in dict_


def test__infer_date_format():
    # Test with an ambiguous date format
    # but with a single format that works for all rows
    date_column = pd.Series(["01-01-2022", "13-01-2022", "01-03-2022"])
    assert _infer_date_format(date_column) == "%d-%m-%Y"

    date_column = pd.Series(["01-01-2022", "01-13-2022", "01-03-2022"])
    assert _infer_date_format(date_column) == "%m-%d-%Y"

    # Test with an ambiguous date format
    # but several formats that work for all rows
    date_column = pd.Series(["01-01-2022", "01-02-2019", "01-03-2019"])
    # check that a warning is raised
    with pytest.warns(UserWarning):
        assert (
            _infer_date_format(date_column) == "%m-%d-%Y"
            or _infer_date_format(date_column) == "%d-%m-%Y"
        )

    # Test with irreconcilable date formats
    date_column = pd.Series(["01-01-2022", "13-01-2019", "01-03-2022", "01-13-2019"])
    assert _infer_date_format(date_column) is None

    # Test previous cases with missing values

    date_column = pd.Series(["01-01-2022", "13-01-2022", "01-03-2022", pd.NA])
    assert _infer_date_format(date_column) == "%d-%m-%Y"

    date_column = pd.Series(["01-01-2022", "01-13-2022", "01-03-2022", pd.NA])
    assert _infer_date_format(date_column) == "%m-%d-%Y"

    date_column = pd.Series(["01-01-2022", "01-02-2019", "01-03-2019", pd.NA])
    # check that a warning is raised
    with pytest.warns(UserWarning):
        assert (
            _infer_date_format(date_column) == "%m-%d-%Y"
            or _infer_date_format(date_column) == "%d-%m-%Y"
        )

    date_column = pd.Series(
        ["01-01-2022", "13-01-2019", "01-03-2022", "01-13-2019", pd.NA]
    )
    assert _infer_date_format(date_column) is None

    # Test previous cases with hours and minutes

    date_column = pd.Series(
        ["01-01-2022 12:00", "13-01-2022 12:00", "01-03-2022 12:00"]
    )
    assert _infer_date_format(date_column) == "%d-%m-%Y %H:%M"

    date_column = pd.Series(
        ["01-01-2022 12:00", "01-13-2022 12:00", "01-03-2022 12:00"]
    )
    assert _infer_date_format(date_column) == "%m-%d-%Y %H:%M"

    date_column = pd.Series(
        ["01-01-2022 12:00", "01-02-2019 12:00", "01-03-2019 12:00"]
    )
    # check that a warning is raised
    with pytest.warns(UserWarning):
        assert (
            _infer_date_format(date_column) == "%m-%d-%Y %H:%M"
            or _infer_date_format(date_column) == "%d-%m-%Y %H:%M"
        )

    date_column = pd.Series(
        ["01-01-2022 12:00", "13-01-2019 12:00", "01-03-2022 12:00", "01-13-2019 12:00"]
    )
    assert _infer_date_format(date_column) is None

    # Test with an empty column
    date_column = pd.Series([], dtype="object")
    assert _infer_date_format(date_column) is None

    # Test with a column containing only NaN values
    date_column = pd.Series([pd.NA, pd.NA, pd.NA])
    assert _infer_date_format(date_column) is None

    # Test with a column containing both dates and non-dates
    date_column = pd.Series(["2022-01-01", "2022-01-02", "not a date"])
    assert _infer_date_format(date_column) is None

    # Test with a column containing more than two date formats
    date_column = pd.Series(["2022-01-01", "01/02/2022", "20220103", "2022-Jan-04"])
    assert _infer_date_format(date_column) is None


test__infer_date_format()
