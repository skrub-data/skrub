import os
import tempfile
import time

from skrub._data_ops import _utils


def test_prune_folder_with_standard_name():
    eight_days_ago = time.time() - 8 * 24 * 3600
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            with open(os.path.join(tmpdir, f"full_data_op_report_{i}.txt"), "w") as f:
                f.write("This is a test file.\n")
            if i < 2:
                os.utime(
                    os.path.join(tmpdir, f"full_data_op_report_{i}.txt"),
                    (eight_days_ago, eight_days_ago),
                )

        assert len(os.listdir(tmpdir)) == 3

        _utils.prune_folder(tmpdir)

        remaining_items = os.listdir(tmpdir)
        assert len(remaining_items) == 1


def test_prune_folder_with_nonstandard_name():
    eight_days_ago = time.time() - 8 * 24 * 3600
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "other_report.txt"), "w") as f:
            f.write("This is a test file.\n")
            os.utime(
                os.path.join(tmpdir, "other_report.txt"),
                (eight_days_ago, eight_days_ago),
            )

        assert len(os.listdir(tmpdir)) == 1

        _utils.prune_folder(tmpdir)

        remaining_items = os.listdir(tmpdir)
        assert len(remaining_items) == 1
