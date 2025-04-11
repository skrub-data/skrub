"""
Get the Wikipedia embeddings for feature augmentation.
"""
# Required for ignoring lines too long in the docstrings
# flake8: noqa: E501

import urllib.request
import warnings
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from urllib.error import URLError

import pandas as pd
from sklearn.decomposition import PCA

from skrub._utils import import_optional_dependency

from ._utils import _sha256, get_data_dir

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

_correspondence_table_url = (
    "https://raw.githubusercontent.com/skrub-data/datasets"
    "/master/data/ken_correspondence.csv"
)


@dataclass(unsafe_hash=True)
class DatasetAll:
    """
    Represents a dataset and its information.
    With this state, the dataset is loaded in memory as a DataFrame (`X` and `y`).
    Additional information such as `path` and `read_csv_kwargs` are provided
    in case the dataframe has to be read from disk, as such:

    .. code:: python

        ds = fetch_employee_salaries(load_dataframe=False)
        df = pd.read_csv(ds.path, **ds.read_csv_kwargs)
    """

    name: str
    description: str
    source: str
    target: str
    X: pd.DataFrame
    y: pd.Series
    path: Path
    read_csv_kwargs: dict[str]

    def __eq__(self, other):
        """
        Implemented for the tests to work without bloating the code.
        The main reason for which it's needed is that equality between
        DataFrame (`X` and `y`) is often ambiguous and will raise an error.
        """
        return (
            self.name == other.name
            and self.description == other.description
            and self.source == other.source
            and self.target == other.target
            and self.X.equals(other.X)
            and self.y.equals(other.y)
            and self.path == other.path
            and self.read_csv_kwargs == other.read_csv_kwargs
        )


@dataclass(unsafe_hash=True)
class DatasetInfoOnly:
    """
    Represents a dataset and its information.
    With this state, the dataset is NOT loaded in memory, but can be read
    with `path` and `read_csv_kwargs`, as such:

    .. code:: python

        ds = fetch_employee_salaries(load_dataframe=False)
        df = pd.read_csv(ds.path, **ds.read_csv_kwargs)
    """

    name: str
    description: str
    source: str
    target: str
    path: Path
    read_csv_kwargs: dict[str]


def fetch_figshare(
    figshare_id: str,
    *,
    target=None,
    load_dataframe=True,
    data_directory=None,
):
    """Fetches a table of from figshare.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="figshare",
        dataset_name=f"figshare_{figshare_id!r}",
        dataset_id=figshare_id,
        target=target,
        load_dataframe=load_dataframe,
        data_directory=data_directory,
    )


def _fetch_dataset_as_dataclass(
    source,
    dataset_name,
    dataset_id,
    target,
    load_dataframe,
    data_directory=None,
    read_csv_kwargs=None,
):
    """Fetches a dataset from a source, and returns it as a dataclass.

    Takes a dataset identifier, a target column name (if applicable),
    and some additional keyword arguments for read_csv.

    If you don't need the dataset to be loaded in memory,
    pass `load_dataframe=False`.

    To save/load the dataset to/from a specific directory,
    pass `data_directory`. If `None`, uses the default skrub
    data directory.

    If the dataset doesn't have a target (unsupervised learning or inapplicable),
    explicitly specify `target=None`.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    if isinstance(data_directory, str):
        data_directory = Path(data_directory)

    if source == "figshare":
        info = _fetch_figshare(dataset_id, data_directory)
    else:
        raise ValueError(f"Unknown source {source!r}")

    if read_csv_kwargs is None:
        read_csv_kwargs = {}

    if target is None:
        target = []

    if load_dataframe:
        if source == "figshare":
            df = pd.read_parquet(info["path"])
        else:
            df = pd.read_csv(info["path"], **read_csv_kwargs)
        y = df[target]
        X = df.drop(target, axis="columns")
        dataset = DatasetAll(
            name=dataset_name,
            description=info["description"],
            source=info["source"],
            target=target,
            X=X,
            y=y,
            path=info["path"],
            read_csv_kwargs=read_csv_kwargs,
        )
    else:
        dataset = DatasetInfoOnly(
            name=dataset_name,
            description=info["description"],
            source=info["source"],
            target=target,
            path=info["path"],
            read_csv_kwargs=read_csv_kwargs,
        )

    return dataset


def _fetch_figshare(
    figshare_id,
    data_directory,
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
    if data_directory is None:
        data_directory = get_data_dir(name="figshare")

    parquet_path = (data_directory / f"figshare_{figshare_id}.parquet").resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    url = f"https://ndownloader.figshare.com/files/{figshare_id}"
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


def fetch_ken_table_aliases():
    """Get the supported aliases of embedded KEN entities tables.

    These aliases can be using in subsequent functions (see section *See Also*).

    Returns
    -------
    set of str
        The aliases of the embedded entities tables.

    See Also
    --------
    fetch_ken_types
        Helper function to search for entity types.
    fetch_ken_embeddings
        Download Wikipedia embeddings by type.

    Notes
    -----
    Requires an Internet connection to work.

    Examples
    --------
    Let's see what are the current KEN subtables available
    for download:

    >>> sorted(fetch_ken_table_aliases())
    ['albums', 'all_entities', 'companies', 'games', 'movies', 'schools']
    """
    correspondence = pd.read_csv(_correspondence_table_url)
    return set(["all_entities"] + list(correspondence["table"].values))


def fetch_ken_types(
    search=None,
    *,
    exclude=None,
    embedding_table_id="all_entities",
):
    """Helper function to search for KEN entity types.

    The result can then be used with fetch_ken_embeddings.

    Parameters
    ----------
    search : str, optional
        Substring pattern that filters the types of entities.
    exclude : str, optional
        Substring pattern to exclude from the search.
    embedding_table_id : str, default='all_entities'
        Table of embedded entities from which to extract the embeddings.
        Get the supported tables with fetch_ken_table_aliases.
        It is NOT possible to pass a custom figshare ID.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        The types of entities containing the substring.

    See Also
    --------
    fetch_ken_embeddings
        Download Wikipedia embeddings by type.

    References
    ----------
    For more details, see Cvetkov-Iliev, A., Allauzen, A. & Varoquaux, G.:
    `Relational data embeddings for feature enrichment
    with background information. <https://doi.org/10.1007/s10994-022-06277-7>`_

    Notes
    -----
    Best used in conjunction with fetch_ken_embeddings.

    This function requires `pyarrow` to be installed.

    Examples
    --------
    To get all the existing KEN types of entities:

    >>> embedding_types = fetch_ken_types()  # doctest: +SKIP
    >>> embedding_types.head() # doctest: +SKIP
                                                    Type
    0                 wikicat_italian_male_screenwriters
    1  wikicat_21st-century_roman_catholic_archbishop...
    2                 wikicat_2000s_romantic_drama_films
    3                  wikicat_music_festivals_in_france
    4        wikicat_20th-century_american_women_artists

    Let's search for all KEN types with the strings "dance" or "music":

    >>> embedding_filtered_types = fetch_ken_types(search="dance|music") # doctest: +SKIP
    >>> embedding_filtered_types.head() # doctest: +SKIP
                                                    Type
    0                  wikicat_music_festivals_in_france
    1  wikicat_films_scored_by_bharadwaj_(music_direc...
    2                  wikicat_english_music_journalists
    3       wikicat_20th-century_american_male_musicians
    4  wikicat_alumni_of_the_london_academy_of_music_...
    """
    correspondence = pd.read_csv(_correspondence_table_url)
    if embedding_table_id not in fetch_ken_table_aliases():
        raise ValueError(
            f"The embedding_table_id must be one of {correspondence['table'].unique()}."
        )
    unique_types_figshare_id = correspondence[
        correspondence["table"] == embedding_table_id
    ]["unique_types_figshare_id"].values[0]
    unique_types = fetch_figshare(unique_types_figshare_id)
    if search is None:
        search_result = unique_types.X
    else:
        search_result = unique_types.X[unique_types.X["Type"].str.contains(search)]
    if exclude is not None:
        search_result = search_result[~search_result["Type"].str.contains(exclude)]
    search_result["Type"] = search_result["Type"].str[1:-1]
    return search_result.reset_index(drop=True)


def fetch_ken_embeddings(
    search_types=None,
    *,
    exclude=None,
    embedding_table_id="all_entities",
    embedding_type_id=None,
    pca_components=None,
    suffix="",
):
    """Download Wikipedia embeddings by type.

    More details on the embeddings can be found on
    https://soda-inria.github.io/ken_embeddings/.

    Parameters
    ----------
    search_types : str, optional
        Substring pattern that filters the types of entities.
        Will keep all entity types containing the substring.
        Write in lowercase. If `None`, all types will be passed.
    exclude : str, optional
        Type of embeddings to exclude from the types search.
    embedding_table_id : str, default='all_entities'
        Table of embedded entities from which to extract the embeddings.
        Get the supported tables with fetch_ken_table_aliases.
        It is also possible to pass a custom figshare ID.
    embedding_type_id : str, optional
        Figshare ID of the file containing the type of embeddings.
        Get the supported tables with fetch_ken_types.
        Ignored unless a custom `embedding_table_id` is provided.
    pca_components : int, optional
        Size of the dimensional space on which the embeddings will be projected
        by a principal component analysis.
        If None, the default dimension (200) of the embeddings will be kept.
    suffix : str, optional, default=''
        Suffix to add to the column names of the embeddings.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        The embeddings of entities and the specified type from Wikipedia.

    See Also
    --------
    fetch_ken_table_aliases :
        Get the supported aliases of embedded entities tables.
    fetch_ken_types :
        Helper function to search for entity types.
    fuzzy_join :
        Join two tables (dataframes) based on approximate column matching.
    Joiner :
        Transformer to enrich a given table via one or more fuzzy joins to
        external resources.

    References
    ----------
    For more details, see Cvetkov-Iliev, A., Allauzen, A. & Varoquaux, G.:
    `Relational data embeddings for feature enrichment
    with background information. <https://doi.org/10.1007/s10994-022-06277-7>`_

    Notes
    -----
    The files are read and returned in parquet format, this function needs
    `pyarrow` installed to run correctly.

    The `search_types` parameter is there to filter the types by the input string
    pattern.
    In case the input is "music", all types with this string will be included
    (e.g. "wikicat_musician_from_france", "wikicat_music_label" etc.).
    Going directly for the exact type name (e.g. "wikicat_rock_music_bands")
    is possible but may not be complete (as some relevant bands may be
    in other similar types).
    For exploring available types, the fetch_ken_types
    function can be used.

    Examples
    --------
    fetch_ken_embeddings allows you to extract embeddings
    you are interested in. For instance, if we are interested in
    video games:

    >>> games_embedding = fetch_ken_embeddings(search_types="video_games") # doctest: +SKIP
    >>> games_embedding.head() # doctest: +SKIP
                             Entity  ...      X199
    0             A_Little_Princess  ...  0.04...
    1                 The_Dark_Half  ... -0.00...
    2                  Frankenstein  ... -0.11...
    3                 Albert_Wesker  ... -0.16...
    4  Harukanaru_Toki_no_Naka_de_3  ...  0.14...

    Extracts all embeddings with the "games" type.
    For the list of existing types see fetch_ken_types.

    Some tables are available pre-filtered for us using the
    `embedding_table_id` parameter:

    >>> games_embedding_fast = fetch_ken_embeddings(embedding_table_id="games") # doctest: +SKIP
    >>> games_embedding_fast.head() # doctest: +SKIP
                         Entity  ...      X199
    0              R-Type_Delta  ...  0.04...
    1  Just_Add_Water_(company)  ... -0.02...
    2                 Li_Xiayan  ...  0.00...
    3             Vampire_Night  ... -0.14...
    4               Shatterhand  ...  0.19...

    It takes less time to load the wanted output, and is more precise as the
    types have been carefully filtered out.
    For a list of pre-filtered tables, see func:`fetch_ken_table_aliases`.
    """
    if embedding_table_id in fetch_ken_table_aliases():
        correspondence = pd.read_csv(_correspondence_table_url)
        embeddings_id = correspondence[correspondence["table"] == embedding_table_id][
            "entities_figshare_id"
        ].values[0]
        embedding_type_id = correspondence[
            correspondence["table"] == embedding_table_id
        ]["type_figshare_id"].values[0]
    else:
        embeddings_id = embedding_table_id
    emb_type = fetch_figshare(embedding_type_id).X
    if search_types is not None:
        emb_type = emb_type[emb_type["Type"].str.contains(search_types)]
    if exclude is not None:
        emb_type = emb_type[~emb_type["Type"].str.contains(exclude)]
    emb_type.drop_duplicates(subset=["Entity"], inplace=True)
    emb_final = []
    emb_full = fetch_figshare(embeddings_id)
    for path in emb_full.path:
        emb_extracts = pd.read_parquet(path)
        emb_extracts = pd.merge(emb_type, emb_extracts, on="Entity")
        emb_extracts.reset_index(drop=True, inplace=True)
        if pca_components is not None:
            pca_i = PCA(n_components=pca_components, random_state=0)
            emb_columns = []
            for j in range(pca_components):
                name = "X" + str(j) + suffix
                emb_columns.append(name)
            pca_embeddings = pca_i.fit_transform(
                emb_extracts.drop(columns=["Entity", "Type"])
            )
            pca_embeddings = pd.DataFrame(pca_embeddings, columns=emb_columns)
            emb_pca = pd.concat(
                [emb_extracts[["Entity", "Type"]], pca_embeddings], axis=1
            )
            emb_final.append(emb_pca)
        else:
            if suffix != "":
                emb_extracts.columns = [
                    col if col in ["Entity", "Type"] else col + suffix
                    for col in emb_extracts.columns
                ]
            emb_final.append(emb_extracts)
    emb_df = pd.concat(emb_final)
    emb_df["Entity"] = emb_df["Entity"].str[1:-1]
    emb_df["Type"] = emb_df["Type"].str[1:-1]
    return emb_df.reset_index(drop=True)
