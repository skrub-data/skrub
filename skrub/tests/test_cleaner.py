import skrub._dataframe as sbd
from skrub import Cleaner
from skrub._to_str import ToStr


def test_add_tostr(df_module):
    """
    Test that the Cleaner conditionally applies the ToStr transformer
    depending on the add_tostr parameter.
    """

    df = df_module.DataFrame({"a": [[1, 2], [3]]})

    # -----------------------
    # Case 1: add_tostr=False
    # -----------------------
    cleaner = Cleaner(add_tostr=False)
    out = cleaner.fit_transform(df)

    # Should preserve dtype
    assert sbd.dtype(out["a"]) == sbd.dtype(df["a"])

    # ----------------------
    # Case 2: add_tostr=True
    # ----------------------
    cleaner = Cleaner(add_tostr=True)
    out = cleaner.fit_transform(df)

    # Apply ToStr manually to a single column (correct usage)
    expected_col = ToStr().fit_transform(df["a"])

    # Compare the dtype of Cleaner output with dtype of ToStr-transformed column
    assert sbd.dtype(out["a"]) == sbd.dtype(expected_col)
