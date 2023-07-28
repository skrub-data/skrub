"""
Private API for dataset fetching.
"""

# Future notes:
# - Watch out for ``fetch_openml()`` API modifications:
# as of january 2021, the function is marked as experimental.

import urllib.request
import warnings
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.error import URLError
from zipfile import BadZipFile, ZipFile

import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.datasets import fetch_openml
from sklearn.datasets._base import _sha256

from skrub._utils import import_optional_dependency, parse_version
from skrub.datasets._utils import get_data_dir

# The IDs of the datasets, from OpenML.
# For each dataset, its URL is constructed as follows:
openml_url: str = "https://www.openml.org/d/{id}"

# A dictionary storing the sha256 hashes of the figshare files
figshare_id_to_hash = {
    39142985: "47d73381ef72b050002a8642194c6718a4954ec9e6c556f4c4ddc6ed84ceec92",
    39149066: "e479cf9741a90c40401697e7fa54409e3b9cfa09f27502877382e64e86fbfcd0",
    39149069: "7b0dcdb15d3aeecba6022c929665ee064f6fb4b8b94186a6e89b6fbc781b3775",
    39149072: "4f58f15168bb8a6cc8b152bd48995bc7d1a4d4d89a9e22d87aa51ccf30118122",
    39149075: "7037603362af1d4bf73551d50644c0957cb91d2b4892e75413f57f415962029a",
    39254360: "531130c714ba6ee9902108d4010f42388aa9c0b3167d124cd57e2c632df3e05a",
    39266300: "37b23b2c37a1f7ff906bc7951cbed4be15d8417dad0762092282f7b491cf8c21",
    39266678: "4e041322985e078de8b08acfd44b93a5ce347c1e501e9d869651e753de747ba1",
    40019230: "4d43fed75dba1e59a5587bf31c1addf2647a1f15ebea66e93177ccda41e18f2f",
    40019788: "67ae86496c8a08c6cc352f573160a094f605e7e0da022eb91c603abb7edf3747",
}


def _resolve_path(path: str | Path | None, suffix: str = "") -> Path:
    """Resolves a path to a file.

    Parameters
    ----------
    path : str or pathlib.Path or None
        The path to resolve.
        If None, uses the default data directory.
    suffix : str
        If ``path`` is unspecified (``None``), adds this suffix to the
        default data directory.

    Returns
    -------
    pathlib.Path
        The resolved path as a pathlib.Path object.
        The directory exists (it was created if necessary).
    """
    if path is None:
        path = get_data_dir(suffix)
    elif isinstance(path, str):
        path = Path(path).expanduser().resolve()
    elif isinstance(path, Path):
        path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_openml_dataset(
    dataset_id: int,
    data_directory: Path,
    target: str,
) -> dict[str, Any]:
    """
    Gets a dataset from OpenML (https://www.openml.org).

    Parameters
    ----------
    dataset_id : int
        The ID of the dataset to fetch.
    target : str
        The name of the target column.
    data_directory : pathlib.Path
        The directory where the dataset is stored.
        By default, a subdirectory "openml" in the skrub data directory.

    Returns
    -------
    mapping of str to any
        A dictionary containing:
          - `description` : str
              The description of the dataset,
              as gathered from OpenML.
          - `source` : str
              The dataset's URL from OpenML.
          - `target` : str
              The name of the target column.
          - `X` : pandas.DataFrame
              The dataset's features.
          - `y` : pandas.Series
              The dataset's target.
    """
    url = openml_url.format(id=dataset_id)

    # The ``fetch_openml()`` function returns a Scikit-Learn ``Bunch`` object,
    # which behaves just like a ``namedtuple``.
    kwargs = {}
    if parse_version("1.2") <= parse_version(sklearn_version) < parse_version("1.2.2"):
        # Avoid the warning, but don't use auto yet because of
        # https://github.com/scikit-learn/scikit-learn/issues/25478
        kwargs.update({"parser": "liac-arff"})
    elif parse_version(sklearn_version) >= parse_version("1.2.2"):
        kwargs.update({"parser": "auto"})

    bunch = fetch_openml(
        data_id=dataset_id,
        data_home=str(data_directory),
        as_frame=True,
        target_column=target,
        **kwargs,
    )

    return {
        "description": bunch.DESCR,
        "source": url,
        "target": target,
        "X": bunch.data,
        "y": bunch.target,
    }


def _fetch_world_bank_data(
    indicator_id: str,
    data_directory: Path,
    download_if_missing: bool,
) -> dict[str, Any]:
    """Gets a dataset from World Bank open data platform (https://data.worldbank.org/).

    Parameters
    ----------
    indicator_id : str
        The ID of the indicator's dataset to fetch.
    data_directory : pathlib.Path
        The directory where the dataset is stored.
        By default, a subdirectory "world_bank" in the skrub data directory.
    download_if_missing : bool
        Whether to download the data if it is not found locally.

    Returns
    -------
    mapping of str to any
        A dictionary containing:
          - `description` : str
              The description of the dataset,
              as gathered from World Bank data.
          - `source` : str
              The dataset's URL from the World Bank data platform.
          - `path` : pathlib.Path
              The local path leading to the dataset,
              saved as a CSV file.
    """
    csv_path = data_directory / f"{indicator_id}.csv"
    url = f"https://api.worldbank.org/v2/en/indicator/{indicator_id}?downloadformat=csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        indicator_name = df.columns[1]
    else:
        if not download_if_missing:
            raise FileNotFoundError(
                f"Couldn't find file for indicator {indicator_id!r} locally. "
            )

        try:
            filehandle, _ = urllib.request.urlretrieve(url)
            zip_file_object = ZipFile(filehandle, "r")
            for name in zip_file_object.namelist():
                if "Metadata" not in name:
                    true_file = name
                    break
            else:
                raise FileNotFoundError(
                    "Could not find any non-metadata file "
                    f"for indicator {indicator_id!r}."
                )
            file = zip_file_object.open(true_file)
        except BadZipFile as e:
            raise FileNotFoundError(
                "Couldn't find csv file, the indicator id "
                f"{indicator_id!r} seems invalid."
            ) from e
        except URLError:
            raise URLError("No internet connection or the website is down.")

        # Read and modify the csv file
        df = pd.read_csv(file, skiprows=3)  # FIXME: why three rows?
        indicator_name = df.iloc[0, 2]
        df[indicator_name] = df.stack().groupby(level=0).last()
        df = df[df[indicator_name] != indicator_id]
        df = df[["Country Name", indicator_name]]

        df.to_csv(csv_path, index=False)

    return {
        "description": (
            f"This table shows the {indicator_name!r} World Bank indicator. "
        ),
        "source": url,
        "X": df,
        "y": None,
    }


def _fetch_figshare(
    figshare_id: str,
    data_directory: Path | None,
    download_if_missing: bool,  # TODO
) -> dict[str, Any]:
    """Fetch a dataset from figshare using the download ID number.

    Parameters
    ----------
    figshare_id : str
        The ID of the dataset to fetch.
    data_directory : pathlib.Path, optional
        The directory where the dataset is stored.
        By default, a subdirectory "figshare" in the skrub data directory.
    download_if_missing : bool
        Whether to download the data if it is not found locally.
        FIXME: Currently not implemented.

    Returns
    -------
    mapping of str to any
        A dictionary containing:
          - `description` : str
              The description of the dataset.
          - `source` : str
              The dataset's URL.
          - `path` : pathlib.Path
              The local path leading to the dataset,
              saved as a parquet file.

    Notes
    -----
    The files are read and returned in parquet format, this function needs
    pyarrow installed to run correctly.
    """
    parquet_path = data_directory.resolve() / f"figshare_{figshare_id}.parquet"
    url = f"https://ndownloader.figshare.com/files/{figshare_id}"
    description = f"This table shows the {figshare_id!r} figshare file."
    existing_files = [
        file
        for file in data_directory.iterdir()
        if figshare_id in file.name and file.suffix == ".parquet"
    ]
    if len(existing_files) > 0:
        if len(existing_files) == 1:
            parquet_paths = [str(existing_files[0].resolve())]
        else:
            parquet_paths = []
            for path in existing_files:
                parquet_path = str(path.resolve())
                parquet_paths += [parquet_path]
        return {
            "description": description,
            "source": url,
            "X": pd.read_parquet(parquet_paths),
            "y": None,
        }
    else:
        warnings.warn(
            f"Could not find the dataset {figshare_id!r} locally. "
            "Downloading it from figshare; this might take a while... "
            "If it is interrupted, some files might be invalid/incomplete: "
            "if on the following run, the fetching raises errors, you can try "
            f"fixing this issue by deleting the directory {parquet_path!s}.",
            UserWarning,
            stacklevel=2,
        )
        import_optional_dependency(
            "pyarrow", extra="pyarrow is required for parquet support."
        )
        from pyarrow.parquet import ParquetFile

        try:
            filehandle, _ = urllib.request.urlretrieve(url)

            # checksum the file
            checksum = _sha256(filehandle)
            if figshare_id in figshare_id_to_hash:
                expected_checksum = figshare_id_to_hash[figshare_id]
                if checksum != expected_checksum:
                    raise OSError(
                        f"{filehandle!r} SHA256 checksum differs from "
                        f"expected ({checksum}!={expected_checksum}) ; "
                        "file is probably corrupted. Please try again. "
                        "If the error persists, please open an issue on GitHub. "
                    )

            df = ParquetFile(filehandle)
            record = df.iter_batches(
                batch_size=1_000_000,
            )
            idx = []
            for x in chain(
                range(0, df.metadata.num_rows, 1_000_000), [df.metadata.num_rows]
            ):
                idx += [x]
            parquet_paths = []
            for i in range(1, len(idx)):
                parquet_path = (
                    data_directory / f"figshare_{figshare_id}_{idx[i]}.parquet"
                )
                batch = next(record).to_pandas()
                batch.to_parquet(parquet_path, index=False)
                parquet_paths += [parquet_path]
            return {
                "description": description,
                "source": url,
                "X": pd.read_parquet(parquet_paths),
                "y": None,
            }
        except URLError:
            raise URLError("No internet connection or the website is down.")
