import pytest

import skrub
from skrub import datasets


def test_output_dir(tmp_path):
    e = skrub.X()
    assert e.skb.full_report(open=False)["report_path"].is_relative_to(
        datasets.get_data_dir()
    )
    out = tmp_path / "report"
    assert (
        e.skb.full_report(open=False, output_dir=out)["report_path"]
        == out / "index.html"
    )
    with pytest.raises(FileExistsError, match=".*Set 'overwrite=True'"):
        e.skb.full_report(open=False, output_dir=out)

    assert (
        e.skb.full_report(open=False, output_dir=out, overwrite=True)["report_path"]
        == out / "index.html"
    )
