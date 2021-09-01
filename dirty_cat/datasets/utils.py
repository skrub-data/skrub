import os
from pathlib import Path


def get_data_dir(name=None) -> Path:
    """
    Returns the directory in which dirty_cat looks for data.

    This is typically useful for the end-user to check where the data is downloaded and stored.
    """
    # assuming we are in datasets.utils, this calls the module
    module_path = Path(os.path.dirname(__file__)).resolve()
    data_dir = module_path / 'data'
    if name is not None:
        data_dir = data_dir / name
    return data_dir
