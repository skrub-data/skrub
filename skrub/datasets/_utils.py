import hashlib
import json
import shutil
import time
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests
from sklearn.utils import Bunch

DATASET_INFO = {
    "medical_charge": {
        "urls": [
            "https://osf.io/download/pu2hq/",
            "https://figshare.com/ndownloader/files/51807752",
        ],
        "sha256": "d10a9d7c0862a8bebe9292ed948df9e6e02cdf4415a8e66306b12578f5f56754",
    },
    "employee_salaries": {
        "urls": [
            "https://osf.io/download/bszkv/",
            "https://figshare.com/ndownloader/files/51807500",
        ],
        "sha256": "1a73268a1a5ce0d376e493737a5fcf0d3f8ffb4cafeca20c7b39381bbc943292",
    },
}


def get_data_home(data_home=None):
    """Returns the path of the skrub data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'skrub_data' in the
    user home folder.

    Alternatively, it can be set programmatically by giving an explicit folder
    path. The '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : pathlib.Path or string, optional
        The path to the skrub data directory. If `None`, the default path
        is `~/skrub_data`.

    Returns
    -------
    data_home : pathlib.Path
        The validated path to the skrub data directory.
    """
    if data_home is None:
        data_home = Path("~").expanduser() / "skrub_data"
    else:
        data_home = Path(data_home)
    data_home = data_home.resolve()
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def get_data_dir(name=None, data_home=None):
    """
    Returns the directory in which skrub looks for data.

    This is typically useful for the end-user to check
    where the data is downloaded and stored.

    Parameters
    ----------
    name : str, optional
        Subdirectory name. If omitted, the root data directory is returned.
    data_home : pathlib.Path or str, optional
        The path to skrub data directory. If `None`, the default path
        is `~/skrub_data`.
    """
    data_dir = get_data_home(data_home)
    if name is not None:
        data_dir = data_dir / name
    return data_dir


def load_simple_dataset(dataset_name, data_home=None):
    bunch = _load_dataset_files(dataset_name, data_home)
    bunch["X"] = bunch.pop(dataset_name)
    if (target := bunch.metadata.get("target", None)) is not None:
        bunch["y"] = bunch["X"][target]
        bunch["X"] = bunch["X"].drop(columns=target)
    return bunch


def _load_dataset_files(dataset_name, data_home):
    """
    skrub_data/
        fraud/
            fraud.tar.gz
            fraud/
                baskets.csv
                products.csv
                metadata.json
    """
    data_home = get_data_home(data_home)
    dataset_dir = data_home / dataset_name
    datafiles_dir = dataset_dir / dataset_name
    datafiles_dir.mkdir(parents=True, exist_ok=True)

    if not datafiles_dir.exists() or not any(datafiles_dir.iterdir()):
        _extract_archive(dataset_dir)

    bunch = Bunch()
    for file_path in datafiles_dir.iterdir():
        if file_path.suffix == ".csv":
            bunch[file_path.stem] = pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            bunch[file_path.stem] = json.loads(file_path.read_text(encoding="utf-8"))

    return bunch


def _extract_archive(dataset_dir):
    dataset_name = dataset_dir.name
    archive_path = dataset_dir / f"{dataset_name}.zip"

    metadata = DATASET_INFO[dataset_name]

    if archive_path.exists():
        expected_checksum = metadata["sha256"]
        checksum = _sha256(archive_path)
        if diff_checksum := expected_checksum != checksum:
            warnings.warn(
                f"SHA256 checksum of existing local file {archive_path.name} "
                f"({checksum}) differs from expected ({expected_checksum}): "
                f"re-downloading from {metadata['urls']} ."
            )

    if not archive_path.exists() or diff_checksum:
        _download_archive(dataset_name, archive_path)

    shutil.unpack_archive(archive_path, dataset_dir, format="zip")


def _download_archive(
    dataset_name,
    archive_path,
    retry=3,
    delay=1,
    timeout=30,
    chunk_size=4096,
):
    metadata = DATASET_INFO[dataset_name]
    error_flag = False

    for idx in range(1, retry + 1):
        for target_url in metadata["urls"]:
            print(
                f"Downloading {dataset_name!r} from {target_url} (retry {idx}/{retry})"
            )
            try:
                error_flag = False
                _stream_download(archive_path, target_url, timeout, chunk_size)
            except Exception as e:
                error_flag = True
                warnings.warn(repr(e), category=FutureWarning)

            if not error_flag:
                if _sha256(archive_path) != metadata["sha256"]:
                    raise OSError(
                        f"File {archive_path.stem!r} checksum verification has failed, "
                        "which means the remote file has been updated.\n"
                        "Please update your skrub version."
                    )
                return
        time.sleep(delay)
        delay *= 5

    else:
        raise OSError(
            f"Can't download the file {dataset_name!r} from urls {metadata['urls']}."
        )


def _stream_download(
    archive_path,
    target_url,
    timeout,
    chunk_size,
):
    # We create a temporary file dedicated to this particular download to avoid
    # conflicts with parallel downloads. If the download is successful, the
    # temporary file is atomically renamed to the final file path (with
    # `shutil.move`). We therefore pass `delete=False` to `NamedTemporaryFile`.
    # Otherwise, garbage collecting temp_file would raise an error when
    # attempting to delete a file that was already renamed. If the download
    # fails, the temporary file is removed manually in the except block.
    temp_file = NamedTemporaryFile(
        mode="wb",
        prefix=archive_path.stem + ".part_",
        dir=archive_path.parent,
        delete=False,
    )

    try:
        temp_file_path = Path(temp_file.name)
        response = requests.get(target_url, timeout=timeout, stream=True)
        for chunk in response.iter_content(chunk_size):
            temp_file.write(chunk)

    except (Exception, KeyboardInterrupt):
        Path(temp_file.name).unlink()
        raise

    # The following renaming is atomic whenever temp_file_path and
    # file_path are on the same filesystem. This should be the case most of
    # the time, but we still use shutil.move instead of os.rename in case
    # they are not.
    shutil.move(temp_file_path, archive_path)


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()
