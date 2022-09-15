import pytest

from dirty_cat._check_dependencies import _check_pack_version


@pytest.mark.parametrize(
    "package_name, required_version, sign",
    [
        ("scikit-learn", "0.22.0", "<="),
        (
            "scikit-learn",
            "0.23.0",
            "==",
        ),
        (
            "scikit-learn",
            "1.3.0",
            ">=",
        ),
    ],
)
def test_check_pack_version_fail(package_name, required_version, sign):
    with pytest.raises(ImportError):
        _check_pack_version("dirty_cat", package_name, required_version, sign)


@pytest.mark.parametrize(
    "package_name, required_version, sign",
    [
        ("scikit-learn", "0.22.0", ">="),
        (
            "scikit-learn",
            "0.23.0",
            "!=",
        ),
        (
            "scikit-learn",
            "1.3.0",
            "<=",
        ),
    ],
)
def test_check_pack_version_pass(package_name, required_version, sign):
    _check_pack_version("dirty_cat", package_name, required_version, sign)
