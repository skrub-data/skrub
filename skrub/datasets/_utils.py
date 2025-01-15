from os import environ
from pathlib import Path
from sys import stderr


DATA_HOME_ENVAR_NAME = "SKRUB_DATA_DIRECTORY"
DATA_HOME_ENVAR = environ.get(DATA_HOME_ENVAR_NAME)

if DATA_HOME_ENVAR and (path := Path(DATA_HOME_ENVAR)).is_absolute():
    DATA_HOME_DEFAULT = path
else:
    DATA_HOME_DEFAULT = Path("~").expanduser() / "skrub_data"


print(f"Setting `DATA_HOME_DEFAULT` to '{DATA_HOME_DEFAULT}'.", file=stderr)


def get_data_home(data_home=None):
    """Returns the path of the skrub data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'skrub_data' in the
    user home folder.

    You can even customize the default data directory by setting in your environment
    the `SKRUB_DATA_DIRECTORY` variable to an *absolute directory path*.

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
    if data_home is not None:
        data_home = Path(data_home)

        # Replace any "~" by the user's home directory
        # https://docs.python.org/3/library/pathlib.html#pathlib.Path.expanduser
        data_home = data_home.expanduser()

        # Resolve relative path to absolute path
        # https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
        data_home = data_home.resolve()
    else:
        data_home = DATA_HOME_DEFAULT

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
