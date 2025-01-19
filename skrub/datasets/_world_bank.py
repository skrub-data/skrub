def fetch_world_bank_indicator(
    indicator_id: str,
    *,
    load_dataframe: bool = True,
    data_directory: Path | str | None = None,
) -> DatasetAll | DatasetInfoOnly:
    """Fetches a dataset of an indicator from the World Bank open data platform.

    Description of the dataset:
        The dataset contains two columns: the indicator value and the
        country names. A list of all available indicators can be found
        at https://data.worldbank.org/indicator.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="world_bank",
        dataset_name=f"World Bank indicator {indicator_id!r}",
        dataset_id=indicator_id,
        target=None,
        load_dataframe=load_dataframe,
        data_directory=data_directory,
    )


def _fetch_world_bank_data(
    indicator_id: str,
    data_directory: Path | None = None,
) -> dict[str, Any]:
    """Gets a dataset from World Bank open data platform (https://data.worldbank.org/).

    Parameters
    ----------
    indicator_id : str
        The ID of the indicator's dataset to fetch.
    data_directory : pathlib.Path, optional
        The directory where the dataset is stored.
        By default, a subdirectory "world_bank" in the skrub data directory.

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
    if data_directory is None:
        data_directory = get_data_dir(name="world_bank")

    csv_path = (data_directory / f"{indicator_id}.csv").resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    url = f"https://api.worldbank.org/v2/en/indicator/{indicator_id}?downloadformat=csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path, nrows=0)
        indicator_name = df.columns[1]
    else:
        warnings.warn(
            (
                f"Could not find the dataset {indicator_id!r} locally. "
                "Downloading it from the World Bank; this might take a while... "
                "If it is interrupted, some files might be invalid/incomplete: "
                "if on the following run, the fetching raises errors, you can try "
                f"fixing this issue by deleting the directory {csv_path!s}."
            ),
            UserWarning,
            stacklevel=2,
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
    description = f"This table shows the {indicator_name!r} World Bank indicator."
    return {
        "dataset_name": indicator_name,
        "description": description,
        "source": url,
        "path": csv_path,
    }
