def fetch_movielens(
    dataset_id: str = "ratings",
    *,
    load_dataframe: bool = True,
    data_directory: Path | None = None,
) -> DatasetAll | DatasetInfoOnly:
    """Fetches a dataset from Movielens.

    Parameters
    ----------
    dataset_id : str
        Either 'ratings' or 'movies'

    Returns
    -------
    :obj:`DatasetAll`
        If `load_dataframe=True`

    :obj:`DatasetInfoOnly`
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="movielens",
        dataset_name="ml-latest-small",
        dataset_id=dataset_id,
        target=None,
        load_dataframe=load_dataframe,
        data_directory=data_directory,
    )


def _fetch_movielens(dataset_id: str, data_directory: Path | None = None) -> dict[str]:
    """Downloads a subset of the Movielens dataset.

    Parameters
    ----------
    data_directory : :obj:`~pathlib.Path`
        The directory in which the data will be saved.
    """
    if data_directory is None:
        data_directory = get_data_dir()

    options = ["movies", "ratings"]
    if dataset_id not in options:
        raise ValueError(f"dataset_id options are {options}, got '{dataset_id}'.")

    zip_directory = Path("ml-latest-small")
    file_path = data_directory / zip_directory / f"{dataset_id}.csv"
    detail_path = data_directory / zip_directory / "README.txt"
    if not file_path.is_file() or not detail_path.is_file():
        # If the details file or the features file don't exist,
        # download the dataset.
        warnings.warn(
            (
                f"Could not find the dataset {dataset_id!r} locally. "
                "Downloading it from MovieLens; this might take a while... "
                "If it is interrupted, some files might be invalid/incomplete: "
                "if on the following run, the fetching raises errors, you can try "
                f"fixing this issue by deleting the directory {data_directory!s}."
            ),
            UserWarning,
            stacklevel=2,
        )
        _download_and_write_movielens_dataset(
            dataset_id,
            data_directory,
            zip_directory,
        )

    description = open(detail_path).read()

    url = MOVIELENS_URL.format(zip_directory=zip_directory)

    return {
        "description": description,
        "source": url,
        "path": Path(data_directory) / zip_directory / f"{dataset_id}.csv",
    }


def _download_and_write_movielens_dataset(dataset_id, data_directory, zip_directory):
    url = MOVIELENS_URL.format(zip_directory=zip_directory)
    tmp_file = None
    try:
        tmp_file, _ = urllib.request.urlretrieve(url)
        data_file = str((zip_directory / f"{dataset_id}.csv").as_posix())
        readme_file = str((zip_directory / "README.txt").as_posix())
        with ZipFile(tmp_file, "r") as zip_file:
            zip_file.extractall(
                data_directory,
                members=[data_file, readme_file],
            )
    except Exception:
        if tmp_file is not None and Path(tmp_file).exists():
            Path(tmp_file).unlink()
        raise
