import csv
from pathlib import Path

try:
    import polars as pl
    import polars.selectors as cs

    POLARS_INSTALLED = True
except ImportError:
    POLARS_INSTALLED = False

import enum

import pandas as pd


class ParseStatus(enum.IntFlag):
    SUCCESS = 0
    FAILED = 1


def _is_binary(file_path):
    """
    Check if a file is binary or text.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is binary, False if it is text.
    """
    try:
        with open(file_path, "rb") as file:
            # Read the first 1024 bytes of the file
            chunk = file.read(1024)
            # Check for the presence of null bytes
            if b"\0" in chunk:
                return True
            # Try to decode the chunk as UTF-8
            chunk.decode("utf-8")
            return False
    except (UnicodeDecodeError, ValueError):
        # If decoding fails, the file is likely binary
        return True


def load_table_paths(path_to_tables: str | Path) -> list:
    """Given `path_to_tables`, load all tables in memory and return them as a
    list.

    Args:
        path_to_tables (str | Path): Path to the tables.

    Returns:
        The list of paths that was found by expanding `path_to_tables`.
    """
    table_list = []
    # path expansion, search for tables
    for table_path in path_to_tables.iterdir():
        table_list.append(Path(table_path))
    return table_list


def get_table_stats(df):
    # TODO: implement using function dispatch
    if isinstance(df, pd.DataFrame):
        return _get_stats_pandas(df)
    elif isinstance(df, pl.DataFrame):
        return _get_stats_polars(df)
    else:
        raise ValueError("Dataframe type not supported.")


def _get_stats_pandas(df):
    stats = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
    }

    stats["n_object_columns"] = len(df.select_dtypes(include=["object"]).columns)
    stats["n_float_columns"] = len(df.select_dtypes(include=["float"]).columns)
    stats["n_numerical_columns"] = len(df.select_dtypes(include=["number"]).columns)

    return stats


def _get_stats_polars(df):
    stats = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
    }
    # get column types using polars selectors
    stats["n_object_columns"] = len(df.select(cs.string()).columns)
    stats["n_float_columns"] = len(df.select(cs.float()).columns)
    stats["n_numerical_columns"] = len(df.select(cs.numeric()).columns)

    return stats


def _detect_csv_delimiter(path_to_file):
    with open(path_to_file, "r") as file:
        sample = file.readline()
        # Create a Sniffer object and detect the delimiter
        sniffer = csv.Sniffer()

        try:
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except csv.Error:
            print("Could not detect the delimiter automatically.")
            print(path_to_file)
            return None


def test_parse_csv(path_to_file, engine):
    # sniff delimiter
    delimiter = _detect_csv_delimiter(path_to_file)
    # try to parse with delimiter
    if engine == "polars":
        try:
            _ = pl.read_csv(path_to_file, separator=delimiter, ignore_errors=True)
            parse_status = ParseStatus.SUCCESS
        except pl.exceptions.ComputeError:
            parse_status = ParseStatus.FAILED
    else:
        # TODO: Implement pandas
        try:
            _ = pd.read_csv(path_to_file, delimiter=delimiter)
            parse_status = ParseStatus.SUCCESS
        except ValueError:
            parse_status = ParseStatus.FAILED
    return parse_status


def test_parse_parquet(path_to_file, engine):
    if engine == "polars":
        try:
            _ = pl.read_parquet(path_to_file)
            parse_status = ParseStatus.SUCCESS
        except pl.exceptions.ComputeError:
            parse_status = ParseStatus.FAILED
    else:
        try:
            _ = pd.read_parquet(path_to_file)
        except ValueError:
            parse_status = ParseStatus.FAILED
    return parse_status


def handle_json(path_to_file):
    raise NotImplementedError()
    # We should not bother with json for now
    pass


def evaluate_file(path_to_file, engine="polars"):
    """
    Evaluates a file's metadata and attempts to parse it based on its extension.
    Parameters
    ----------
    path_to_file : str or Path
        The path to the file to be evaluated.
    engine : str, optional
        The parsing engine to use. Defaults to "polars". If "polars" is selected
        but the required library is not installed, a RuntimeError is raised.
    Returns
    -------
    dict
        A dictionary containing the following metadata about the file:
        - is_symlink (bool): Whether the file is a symbolic link.
        - is_dir (bool): Whether the path is a directory.
        - absolute_path (str): The absolute path to the file.
        - filename (str): The name of the file.
        - parent (str): The parent directory of the file.
        - extension (str): The file's extension.
        - binary (bool): Whether the file is binary.
        - file_size (float): The size of the file in kilobytes (KB).
        - parse_status (ParseStatus): The status of parsing the file.
    Raises
    ------
    RuntimeError
        If the "polars" engine is selected but the required library is not installed.
    Notes
    -----
    - The function attempts to determine the file type based on its extension.
    - Supported extensions include ".parquet" and ".csv". Other extensions will
      result in a parse status of `ParseStatus.FAILED`.
    - The `is_binary` function is used to determine if the file is binary.
    - The `test_parse_parquet` and `test_parse_csv` functions are used to test
      parsing for ".parquet" and ".csv" files, respectively.
    """
    if engine == "polars" and not POLARS_INSTALLED:
        raise RuntimeError("Importing polars failed.")

    path_to_file = Path(path_to_file)
    file_info = {
        "is_symlink": False,
        "is_dir": False,
        "absolute_path": str(path_to_file),
        "filename": path_to_file.name,
        "parent": str(path_to_file.parent),
        "extension": "",
        "binary": False,
        "file_size": 0,
        "parse_status": ParseStatus.FAILED,
    }

    file_info["absolute_path"] = str(path_to_file)
    if path_to_file.is_symlink():
        file_info["is_symlink"] = True
    if path_to_file.is_dir():
        file_info["is_dir"] = True
    else:
        # file isn't symlink
        # Settings size to kbytes
        file_info["file_size"] = path_to_file.stat().st_size / 1024
        # trying to guess type
        ext = path_to_file.suffix
        file_info["extension"] = ext
        file_info["binary"] = _is_binary(path_to_file)

        if ext == ".parquet":
            # handle parquet
            parse_status = test_parse_parquet(path_to_file, engine)
        elif ext == ".csv":
            parse_status = test_parse_csv(path_to_file, engine)
        else:
            parse_status = ParseStatus.FAILED

        file_info["parse_status"] = parse_status

    return file_info


def explore_folder(path_to_tables, engine="polars"):
    """Given a path to a collection of tables, iterate over each of them and gather
    information.

    Parameters
    ----------
    path_to_tables : str or Path
        Location of the tables to explore.

    engine: str, default="polars"
        Dataframe engine to use to parse the data. Defaults to polars if installed.
    """
    if engine == "polars" and not POLARS_INSTALLED:
        raise RuntimeError("Importing polars failed.")
    if engine not in ["polars", "pandas"]:
        raise ValueError(f"Dataframe engine {engine} not supported.")

    path_to_tables = Path(path_to_tables).absolute()

    if not path_to_tables.exists():
        raise FileNotFoundError(f"Path {path_to_tables} does not exist.")
    table_list = load_table_paths(path_to_tables)

    stats = []
    # evaluate the statistics for every table
    for f in table_list:
        stats.append(evaluate_file(f, engine=engine))

    return stats


def get_stats_on_tables(df_files: pl.DataFrame, engine="polars"):
    if engine == "polars" and not POLARS_INSTALLED:
        raise RuntimeError("Importing polars failed.")
    if engine not in ["polars", "pandas"]:
        raise ValueError(f"Dataframe engine {engine} not supported.")

    stats = []
    # evaluate the statistics for every table
    for f_row in df_files.filter(parse_status=0).iter_rows(named=True):
        ext = f_row["extension"]
        if ext == ".csv":
            if engine == "polars":
                df = pl.read_csv(f_row["absolute_path"], ignore_errors=True)
            else:
                df = pd.read_csv(f_row["absolute_path"])
        elif ext == ".parquet":
            if engine == "polars":
                df = pl.read_parquet(
                    f_row["absolute_path"],
                )
            else:
                df = pd.read_parquet(f_row["absolute_path"])
        else:
            raise ValueError(f"Extension {ext} not supported.")

        f_path = Path(f_row["absolute_path"])
        f_name = f_path.name
        f_parent = f_path.parent
        stats_this = {"file_name": f_name, "parent": f_parent, "extension": ext}
        stats_this.update(get_table_stats(df))
        stats.append(stats_this)

    return pl.DataFrame(stats)


def find_candidates(table, path_to_tables):
    """Given a dataframe and a path to a collection of tables, find tables in the
    collection that may be joined on the table and rank them by similarity.

    Parameters
    ----------
    table : pandas.DataFrame or polars.DataFrame
        Table to be examined
    path_to_tables : str or Path
        Location of the tables.
    """
    pass
