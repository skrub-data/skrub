import os
from pathlib import Path


def get_data_dir(name: str | None = None) -> Path:
    """
    Returns the directory in which skrub looks for data.

    This is typically useful for the end-user to check
    where the data is downloaded and stored.

    Parameters
    ----------
    name: str, optional
        Subdirectory name. If omitted, the root data directory is returned.
    """
    # Note: we stick to os.path instead of pathlib.Path because
    # it's easier to test, and the functionality is the same.
    module_path = Path(os.path.dirname(__file__)).resolve()
    data_dir = module_path / "data"
    if name is not None:
        data_dir = data_dir / name
    return data_dir


def import_parquet_file_reader():
    """Import `ParquetFile` from either pyarrow or fastparquet.

    Since `pyarrow` and `fastparquet` are optional dependencies,
    we need to import `ParquetFile` dynamically.
    """

    try:
        from pyarrow.parquet import ParquetFile
    except ModuleNotFoundError:
        try:
            from fastparquet import ParquetFile
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "You need to install either pyarrow or fastparquet to use this function."
            )
    return ParquetFile


def is_parquet_file_reader_installed():
    """Check if a `ParquetFile` reader is installed."""
    try:
        import_parquet_file_reader()
        return True
    except ModuleNotFoundError:
        return False
