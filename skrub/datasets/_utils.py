import json
import hashlib
import pandas as pd
import shutil
import time
import warnings
import requests
from pathlib import Path
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


def load_dataset(dataset_name, data_home=None):
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
    
    if not datafiles_dir.exists() or not any(datafiles_dir.iterdir()):
        extract_archive(dataset_dir)
    
    bunch = Bunch()
    for file_path in dataset_dir.iterdir():        
        if file_path.suffix == ".csv":
            bunch[file_path.stem] = pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            metadata_key = f"{file_path.stem}_metadata"
            bunch[metadata_key] = json.loads(file_path.read_text(), "utf-8")
        
    return bunch
    

def extract_archive(dataset_dir):

    dataset_name = dataset_dir.name
    archive_path = dataset_dir / f"{dataset_name}.zip"
    if not archive_path.exists():
        download_archive(dataset_name, archive_path)

    datafiles_dir = dataset_dir / dataset_name
    shutil.unpack_archive(archive_path, datafiles_dir, format="zip")


def download_archive(dataset_name, archive_path, retry=3, delay=1, timeout=30):
    
    metadata = DATASET_INFO[dataset_name]
    error_flag = False

    while True:
        for target_url in metadata["urls"]:
            r = requests.get(target_url, timeout=timeout)
            try:
                error_flag = False
                r.raise_for_status()
                break
            except requests.HTTPError as e:
                error_flag = True
                warnings.warn(e)

        if not error_flag:
            if hashlib.sha256(r.content).hexdigest() != metadata["sha256"]:
                raise OSError(
                    "The file has been updated, please update your skrub version."
                )
            break
        
        if not retry:
            raise OSError(
                f"Can't download the file {dataset_name} from urls {metadata['urls']}."
            )

        time.sleep(delay)
        retry -= 1
        timeout *= 2

    archive_path.write_bytes(r.content)
