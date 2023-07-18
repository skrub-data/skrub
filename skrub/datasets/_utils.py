import os
from pathlib import Path


def get_data_home(data_home=None) -> str:
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
    data_home : str, default=None
        The path to skrub data directory. If `None`, the default path
        is `~/skrub_data`.

    Returns
    -------
    data_home: str or path-like, default=None
        The path to skrub data directory.
    """
    if data_home is None:
        data_home = Path("~") / "skrub_data"
    data_home.make_dir(exist_ok=True)
    return data_home


def get_data_dir(name: str | None = None, data_home: str | None = None) -> Path:
    """
    Returns the directory in which skrub looks for data.

    This is typically useful for the end-user to check
    where the data is downloaded and stored.

    Parameters
    ----------
    data_home : str, default=None
        The path to skrub data directory. If `None`, the default path
        is `~/skrub_data`.
    name: str, optional
        Subdirectory name. If omitted, the root data directory is returned.
    """
    data_home = get_data_home(data_home)
    data_dir = Path(data_home).resolve()
    if name is not None:
        data_dir = data_dir / name
    return data_dir
