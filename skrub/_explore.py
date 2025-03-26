import csv
from pathlib import Path

try:
    import polars as pl

    POLARS_INSTALLED = True
except ImportError:
    POLARS_INSTALLED = False

import pandas as pd


def is_binary(file_path):
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
            # Check for the presence of null bytes, which are common in binary files
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


def _get_stats_pandas(df):
    stats = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
    }

    object_cols = df.select_dtypes(include=["object"]).columns
    stats["n_object_columns"] = len(object_cols)
    float_cols = df.select_dtypes(include=["float"]).columns
    stats["n_float_columns"] = len(float_cols)
    numerical_cols = df.select_dtypes(include=["number"]).columns
    stats["n_numerical_columns"] = len(numerical_cols)

    return stats


def _get_stats_polars(df):
    stats = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
    }

    object_cols = df.select_dtypes(str).columns
    stats["n_object_columns"] = len(object_cols)
    float_cols = df.select_dtypes(float).columns
    stats["n_float_columns"] = len(float_cols)
    numerical_cols = df.select_dtypes([float, int]).columns
    stats["n_numerical_columns"] = len(numerical_cols)

    return stats


def detect_csv_delimiter(path_to_file):
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


def handle_csv(path_to_file, engine):
    # sniff delimiter
    delimiter = detect_csv_delimiter(path_to_file)
    # try to parse with delimiter
    if engine == "polars":
        df = pl.read_csv(path_to_file, separator=delimiter)

        statistics = _get_stats_polars(df)
    else:
        df = pd.read_csv(path_to_file, delimiter=delimiter)
        statistics = _get_stats_pandas(df)
    return statistics


def handle_parquet(path_to_file, engine):
    if engine == "polars":
        df = pl.scan_parquet(path_to_file)
        statistics = _get_stats_polars(df)
    else:
        df = pd.read_parquet(path_to_file)
        statistics = _get_stats_pandas(df)
    return statistics


def handle_json(path_to_file):
    raise NotImplementedError()
    # We should not bother with json for now
    pass


def evaluate_file(path_to_file, engine="polars"):
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
        file_info["binary"] = is_binary(path_to_file)

        if ext == ".parquet":
            # handle parquet
            file_stats = handle_parquet(path_to_file, engine)
        elif ext == ".csv":
            file_stats = handle_csv(path_to_file, engine)
        else:
            # extension is not recognized, do something
            file_stats = ""

        file_info["file_stats"] = file_stats
    return file_info


def explore_data_lake(path_to_tables, engine="polars"):
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

    # overall statistics
    # overall_statistics = {
    #     "n_files": 0,
    #     "n_failed_parses": 0,
    #     "avg_n_rows": 0,
    #     "avg_n_cols": 0,
    # }
    stats = []
    # evaluate the statistics for every table
    for f in table_list:
        stats.append(evaluate_file(f, engine=engine))

    return stats


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
