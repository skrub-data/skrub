import urllib.request
import warnings
from itertools import chain
from urllib.error import URLError

from .._utils import import_optional_dependency
from ._utils import get_data_dir

FIGSHARE_URL = "https://ndownloader.figshare.com/files/{figshare_id}"


def fetch_figshare(
    figshare_id,
    data_directory=None,
):
    """Fetch a dataset from figshare using the download ID number.

    Parameters
    ----------
    figshare_id : str
        The ID of the dataset to fetch.
    data_directory : pathlib.Path, optional
        The directory where the dataset is stored.
        By default, a subdirectory "figshare" in the skrub data directory.

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
    data_directory = data_directory or get_data_dir(name="figshare")

    parquet_path = (data_directory / f"figshare_{figshare_id}.parquet").resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    url = FIGSHARE_URL.format(figshare_id=figshare_id)

    description = f"This table shows the {figshare_id!r} figshare file."
    file_paths = [
        file
        for file in data_directory.iterdir()
        if file.name.startswith(f"figshare_{figshare_id}")
    ]
    if len(file_paths) > 0:
        if len(file_paths) == 1:
            parquet_paths = [str(file_paths[0].resolve())]
        else:
            parquet_paths = []
            for path in file_paths:
                parquet_path = str(path.resolve())
                parquet_paths += [parquet_path]
        return {
            "dataset_name": figshare_id,
            "description": description,
            "source": url,
            "path": parquet_paths,
        }
    else:
        warnings.warn(
            (
                f"Could not find the dataset {figshare_id!r} locally. "
                "Downloading it from figshare; this might take a while... "
                "If it is interrupted, some files might be invalid/incomplete: "
                "if on the following run, the fetching raises errors, you can try "
                f"fixing this issue by deleting the directory {parquet_path!s}."
            ),
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
            for _, x in enumerate(
                chain(range(0, df.metadata.num_rows, 1_000_000), [df.metadata.num_rows])
            ):
                idx += [x]
            parquet_paths = []
            for i in range(1, len(idx)):
                parquet_path = (
                    data_directory / f"figshare_{figshare_id}_{idx[i]}.parquet"
                ).resolve()
                batch = next(record).to_pandas()
                batch.to_parquet(parquet_path, index=False)
                parquet_paths += [parquet_path]
            return {
                "dataset_name": figshare_id,
                "description": description,
                "source": url,
                "path": parquet_paths,
            }
        except URLError:
            raise URLError("No internet connection or the website is down.")
