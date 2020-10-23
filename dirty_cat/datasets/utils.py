import os


def get_data_dir(name=None):
    """
    Returns the directory in which dirty_cat looks for data.

    This is typically useful for the end-user to check where the data is
    downloaded and stored.
    """
    # assuming we are in datasets.utils, this calls the module
    module_path = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(module_path, 'data')
    if name is not None:
        data_dir = os.path.join(data_dir, name)
    return data_dir


def _check_if_exists(path, remove=False):
    if remove:
        try:
            os.remove(path)
        except OSError:
            pass
        return False
    else:
        return os.path.exists(path)
