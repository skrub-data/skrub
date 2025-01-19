import hashlib
import os
import shutil
import time
import warnings
from collections import namedtuple
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlretrieve

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])


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


# Vendored from scikit-learn 1.6.0
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


# Vendored from scikit-learn 1.6.0
def _fetch_remote(remote, dirname=None, n_retries=3, delay=1):
    """Helper function to download a remote dataset.

    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the SHA256 checksum of the
    downloaded file.

    .. versionchanged:: 1.6

        If the file already exists locally and the SHA256 checksums match, the
        path to the local file is returned without re-downloading.

    Parameters
    ----------
    remote : RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum.

    dirname : str or Path, default=None
        Directory to save the file to. If None, the current working directory
        is used.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    delay : int, default=1
        Number of seconds between retries.

        .. versionadded:: 1.5

    Returns
    -------
    file_path: Path
        Full path of the created file.
    """
    if dirname is None:
        folder_path = Path(".")
    else:
        folder_path = Path(dirname)

    file_path = folder_path / remote.filename

    if file_path.exists():
        if remote.checksum is None:
            return file_path

        checksum = _sha256(file_path)
        if checksum == remote.checksum:
            return file_path
        else:
            warnings.warn(
                f"SHA256 checksum of existing local file {file_path.name} "
                f"({checksum}) differs from expected ({remote.checksum}): "
                f"re-downloading from {remote.url} ."
            )

    # We create a temporary file dedicated to this particular download to avoid
    # conflicts with parallel downloads. If the download is successful, the
    # temporary file is atomically renamed to the final file path (with
    # `shutil.move`). We therefore pass `delete=False` to `NamedTemporaryFile`.
    # Otherwise, garbage collecting temp_file would raise an error when
    # attempting to delete a file that was already renamed. If the download
    # fails or the result does not match the expected SHA256 digest, the
    # temporary file is removed manually in the except block.
    temp_file = NamedTemporaryFile(
        prefix=remote.filename + ".part_", dir=folder_path, delete=False
    )
    # Note that Python 3.12's `delete_on_close=True` is ignored as we set
    # `delete=False` explicitly. So after this line the empty temporary file still
    # exists on disk to make sure that it's uniquely reserved for this specific call of
    # `_fetch_remote` and therefore it protects against any corruption by parallel
    # calls.
    temp_file.close()
    try:
        temp_file_path = Path(temp_file.name)
        while True:
            try:
                urlretrieve(remote.url, temp_file_path)
                break
            except (URLError, TimeoutError):
                if n_retries == 0:
                    # If no more retries are left, re-raise the caught exception.
                    raise
                warnings.warn(f"Retry downloading from url: {remote.url}")
                n_retries -= 1
                time.sleep(delay)

        checksum = _sha256(temp_file_path)
        if remote.checksum is not None and remote.checksum != checksum:
            raise OSError(
                f"The SHA256 checksum of {remote.filename} ({checksum}) "
                f"differs from expected ({remote.checksum})."
            )
    except (Exception, KeyboardInterrupt):
        os.unlink(temp_file.name)
        raise

    # The following renaming is atomic whenever temp_file_path and
    # file_path are on the same filesystem. This should be the case most of
    # the time, but we still use shutil.move instead of os.rename in case
    # they are not.
    shutil.move(temp_file_path, file_path)

    return file_path
