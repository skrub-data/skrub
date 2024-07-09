import pytest

from skrub._reporting._cli import make_report
from skrub._reporting._utils import read


@pytest.mark.parametrize("extension", ["parquet", "csv"])
def test_make_report(
    extension,
    tmp_path,
    skip_if_polars_and_numpy2_installed,
    browser_mock,
    data_directory,
):
    data_file = data_directory / f"air_quality_small.{extension}"
    try:
        read(data_file)
    except ImportError:
        assert extension == "parquet"
        pytest.skip("missing pyarrow, cannot read parquet")
    output_file = tmp_path / "report.html"

    # writing to file
    make_report([str(data_file), "-o", str(output_file)])
    report = output_file.read_text("utf-8")
    assert "skrub-table-report" in report
    assert data_file.name in report

    # opening in browser
    args = [str(data_file)]
    if extension == "parquet":
        args += ["--order_by", "date.utc"]
    make_report(args)
    report = browser_mock.content.decode("utf-8")
    assert "skrub-table-report" in report
    assert data_file.name in report
