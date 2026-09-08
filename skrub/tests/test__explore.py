from pathlib import Path

import pytest

from skrub._explore import ParseStatus, try_parse_csv


@pytest.fixture
def csv_file(tmp_path):
    """Fixture to create a temporary CSV file."""
    file_path = tmp_path / "test.csv"
    with open(file_path, "w") as f:
        f.write("col1,col2,col3\n1,2,3\n4,5,6\n")
    return file_path


@pytest.fixture
def invalid_csv_file(tmp_path):
    """Fixture to create a temporary invalid CSV file."""
    file_path = tmp_path / "invalid.csv"
    with open(file_path, "w") as f:
        f.write("col1|col2|col3\n1,2,3\n4|5|6\n")
    return file_path


def test_test_parse_csv_with_polars(csv_file):
    """Test `test_parse_csv` with a valid CSV file using Polars."""
    result = try_parse_csv(csv_file, engine="polars")
    assert result == ParseStatus.SUCCESS


def test_test_parse_csv_with_pandas(csv_file):
    """Test `test_parse_csv` with a valid CSV file using Pandas."""
    result = try_parse_csv(csv_file, engine="pandas")
    assert result == ParseStatus.SUCCESS


def test_test_parse_csv_with_invalid_file_polars(invalid_csv_file):
    """Test `test_parse_csv` with an invalid CSV file using Polars."""
    result = try_parse_csv(invalid_csv_file, engine="polars")
    assert result == ParseStatus.FAILED


def test_test_parse_csv_with_invalid_file_pandas(invalid_csv_file):
    """Test `test_parse_csv` with an invalid CSV file using Pandas."""
    result = try_parse_csv(invalid_csv_file, engine="pandas")
    assert result == ParseStatus.FAILED


def test_test_parse_csv_with_nonexistent_file():
    """Test `test_parse_csv` with a nonexistent file."""
    nonexistent_file = Path("nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        try_parse_csv(nonexistent_file, engine="polars")
