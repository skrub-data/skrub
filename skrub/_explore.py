import csv
import enum
from pathlib import Path

import pandas as pd

from . import _dataframe as sbd
from . import _selectors as s

try:
    import polars as pl
    import polars.selectors as cs

    POLARS_INSTALLED = True
except ImportError:
    POLARS_INSTALLED = False


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


def load_table_paths(path_to_tables):
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
            # TODO: add a better sniffer
            delimiter = sniffer.sniff(sample, delimiters=list(",;^|\t")).delimiter
            return delimiter
        except csv.Error:
            print("Could not detect the delimiter automatically.")
            print(path_to_file)
            return None


def try_parse_csv(path_to_file, engine):
    # sniff delimiter
    delimiter = _detect_csv_delimiter(path_to_file)
    if delimiter is None:
        return ParseStatus.FAILED
    if engine == "polars":
        try:
            _ = pl.read_csv(path_to_file, separator=delimiter, ignore_errors=True)
            parse_status = ParseStatus.SUCCESS
        except pl.exceptions.ComputeError:
            parse_status = ParseStatus.FAILED
    else:
        try:
            _ = pd.read_csv(path_to_file, delimiter=delimiter)
            parse_status = ParseStatus.SUCCESS
        except ValueError:
            parse_status = ParseStatus.FAILED
    return parse_status


def try_parse_parquet(path_to_file, engine):
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
            parse_status = try_parse_parquet(path_to_file, engine)
        elif ext == ".csv":
            parse_status = try_parse_csv(path_to_file, engine)
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


def _read_table(table_path, engine="polars"):
    f_path = Path(table_path)
    ext = f_path.suffix
    if ext == ".csv":
        if engine == "polars":
            df = pl.read_csv(table_path["absolute_path"], ignore_errors=True)
        else:
            df = pd.read_csv(table_path["absolute_path"])
    elif ext == ".parquet":
        if engine == "polars":
            df = pl.read_parquet(
                table_path["absolute_path"],
            )
        else:
            df = pd.read_parquet(table_path["absolute_path"])
    else:
        return None
    return df


def get_stats_on_tables(df_files, engine="polars"):
    if engine == "polars" and not POLARS_INSTALLED:
        raise RuntimeError("Importing polars failed.")
    if engine not in ["polars", "pandas"]:
        raise ValueError(f"Dataframe engine {engine} not supported.")

    # Get the files that have been parsed correctly.
    parsed_files = sbd.filter(df_files, sbd.to_bool(df_files["parse_status"]))

    stats = []
    # evaluate the statistics for every table
    for f_row in sbd.to_list(parsed_files):
        f_path = Path(f_row)
        df = _read_table(f_path, engine)
        ext = f_path.suffix
        f_name = f_path.name
        f_parent = f_path.parent
        stats_this = {"file_name": f_name, "parent": f_parent, "extension": ext}
        stats_this.update(get_table_stats(df))
        stats.append(stats_this)

    return pl.DataFrame(stats)


def find_candidates(table, path_to_tables, engine="polars"):
    """Given a dataframe and a path to a collection of tables, find tables in the
    collection that may be joined on the table and rank them by similarity.

    Parameters
    ----------
    table : pandas.DataFrame or polars.DataFrame
        Table to be examined
    path_to_tables : str or Path
        Location of the tables.
    """
    # TODO: I copypasted this, needs fixing
    raise NotImplementedError()
    # load list of tables
    # find unique values for each table
    # for table_path in path_to_tables:V
    #     table = _read_table(table_path, engine)
    #     _unique_values_candidates[table_path] = find_unique_values(table)

    # # find unique values for the query columns
    # unique_values_X = find_unique_values(X, query_columns)
    # # measure containment
    # containment_list = measure_containment_tables(
    #     unique_values_X, _unique_values_candidates
    # )
    # # prepare ranking
    # _ranking = prepare_ranking(containment_list, budget=budget)


def find_unique_values(table, columns):
    """Given a dataframe and either a list of columns or None, find the unique
    values in each column in the list or all columns, then return the list of values
    as a dictionary in the format {column_name: [list_of_values]}

    Args:
        table: Table to evaluate.
        columns: List of columns to evaluate. If None,
        consider all columns.

    Returns:
        A dict that contains the unique values found for each selected column.
    """
    column_names = sbd.column_names(table)
    # select the columns of interest
    if columns is not None:
        # error checking columns
        if len(columns) == 0:
            raise ValueError("No columns provided.")
        for col in columns:
            if col not in column_names:
                raise ValueError(
                    f"Column {col} not found in table columns {column_names}."
                )
    else:
        # Selecting only columns with strings
        # TODO: string? categorical? both?
        columns = s.string().expand(table)
    unique_values = {}
    # find the unique values
    for col in columns:
        unique_values[col] = sbd.to_numpy(sbd.unique(sbd.col(table, col)))
    # return the dictionary of unique values
    return unique_values


def measure_containment_tables(unique_values_base, unique_values_candidate):
    """Given `unique_values_base` and `unique_values_candidate`, measure the
    containment for each pair.

    The result will be returned as a dataframe with columns "query_column",
    "cand_path", "cand_column", "containment".

    Args:
        unique_values_base (dict): Dictionary that contains the set of unique
        values for each column in the base (query) table.
        unique_values_candidate (dict): Dictionary that contains the set of
        unique values for each column in the candidate table.

    Returns:
        A dataframe that contains each candidate and the corresponding containment.
    """
    containment_list = []
    # TODO: this should absolutely get optimized
    # for each value in unique_values_base, measure the containment for every
    # value in unique_values_candidate
    for path, dict_cand in unique_values_candidate.items():
        for col_base, values_base in unique_values_base.items():
            for col_cand, values_cand in dict_cand.items():
                containment = measure_containment(values_base, values_cand)
                if containment > 0:
                    tup = (col_base, str(path), col_cand, containment)
                    containment_list.append(tup)
    return containment_list


def measure_containment(unique_values_query, unique_values_candidate):
    """Given `unique_values_query` and `unique_values_candidate`, measure the
    Jaccard Containment of the query in the
    candidate column. Return only the containment

    Args:
        unique_values_query (set): Set of unique values in the query.
        unique_values_candidate (set): Set of unique values in the candidate
        column.
    """
    # measure containment
    set_query = set(unique_values_query)
    containment = len(set_query.intersection(set(unique_values_candidate))) / len(
        set_query
    )
    # return containment
    return containment


def prepare_ranking(containment_list, budget):
    """Sort the containment list and cut all candidates past a certain budget.

    Args:
        containment_list (list[tuple]): List of candidates with format
        (query_column, cand_table, cand_column, similarity).
        budget (int): Number of candidates to keep from the list.
    """

    # Sort the list
    containment_list = sorted(containment_list, key=lambda x: x[3], reverse=True)

    # TODO: Somewhere here we might want to do some fancy filtering of the
    # candidates in the ranking (with profiling)

    # Return `budget` candidates
    ranking = containment_list[:budget]
    return ranking
