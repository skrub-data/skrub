import json
import hashlib
import pandas as pd
import os
import shutil
import time
import tarfile
import warnings
import requests
from collections import namedtuple
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlretrieve
from sklearn.utils import Bunch

ARCHIVE_METADATA = {
    "medical_charge": {
        "urls": ["https://figshare.com/ndownloader/files/51807752"],
        "sha256": "d10a9d7c0862a8bebe9292ed948df9e6e02cdf4415a8e66306b12578f5f56754",
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
        datasets/
            fraud/
                baskets.csv
                products.csv
                metadata.json
        archives/
            fraud.tar.gz            
    """
    data_home = get_data_home(data_home)
    dataset_dir = data_home / "datasets" / dataset_name
    
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        extract_archive(dataset_name, data_home)
    
    bunch = Bunch()
    for file_path in dataset_dir.iterdir():        
        if file_path.suffix == ".csv":
            bunch[file_path.stem] = pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            metadata_key = f"{file_path.stem}_metadata"
            bunch[metadata_key] = json.loads(file_path.read_text(), "utf-8")
        
    return bunch
    

def extract_archive(dataset_name, data_home):

    archive_path = data_home / "archives" / dataset_name
    if not archive_path.exists():
        download_archive(dataset_name, data_home)

    dataset_dir = data_home / "datasets"
    shutil.unpack_archive(archive_path, dataset_dir, format="zip")


def download_archive(dataset_name, data_home, retry=3, delay=1, timeout=30):
    
    metadata = ARCHIVE_METADATA[dataset_name]
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
        
        if hashlib.sha256(r.content).hexdigest() != metadata["sha256"]:
            raise OSError(
                "The file has been updated, please update your skrub version."
            )

        if not error_flag or not retry:
            break
        time.sleep(delay)
        retry -= 1
        timeout *= 2

    archive_path = data_home / "archives" / dataset_name
    archive_path.write_bytes(r.content) 
