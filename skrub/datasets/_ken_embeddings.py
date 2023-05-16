"""
Get the Wikipedia embeddings for feature augmentation.
"""
from typing import Optional, Set

import pandas as pd
from sklearn.decomposition import PCA

from skrub.datasets import fetch_figshare

# Required for ignoring lines too long in the docstrings
# flake8: noqa: E501


_correspondence_table_url = (
    "https://raw.githubusercontent.com/dirty-cat/datasets"
    "/master/data/ken_correspondence.csv"
)


def get_ken_table_aliases() -> Set[str]:
    """Get the supported aliases of embedded KEN entities tables.

    These aliases can be using in subsequent functions (see section *See Also*).

    Returns
    -------
    set of str
        The aliases of the embedded entities tables.

    See Also
    --------
    :func:`get_ken_types`
        Helper function to search for entity types.
    :func:`get_ken_embeddings`
        Download Wikipedia embeddings by type.

    Notes
    -----
    Requires an Internet connection to work.
    """
    correspondence = pd.read_csv(_correspondence_table_url)
    return set(["all_entities"] + list(correspondence["table"].values))


def get_ken_types(
    search: str = None,
    *,
    exclude: Optional[str] = None,
    embedding_table_id: str = "all_entities",
) -> pd.DataFrame:
    """Helper function to search for KEN entity types.

    The result can then be used with :func:`get_ken_embeddings`.

    Parameters
    ----------
    search : str, optional
        Substring pattern that filters the types of entities.
    exclude : str, optional
        Substring pattern to exclude from the search.
    embedding_table_id : str, default='all_entities'
        Table of embedded entities from which to extract the embeddings.
        Get the supported tables with :func:`get_ken_table_aliases`.
        It is NOT possible to pass a custom figshare ID.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        The types of entities containing the substring.

    See Also
    --------
    :func:`get_ken_embeddings`
        Download Wikipedia embeddings by type.

    References
    ----------
    For more details, see Cvetkov-Iliev, A., Allauzen, A. & Varoquaux, G.:
    `Relational data embeddings for feature enrichment
    with background information. <https://doi.org/10.1007/s10994-022-06277-7>`_

    Notes
    -----
    Best used in conjunction with :func:`get_ken_embeddings`.
    """
    correspondence = pd.read_csv(_correspondence_table_url)
    if embedding_table_id not in get_ken_table_aliases():
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
    return search_result


def get_ken_embeddings(
    types: Optional[str] = None,
    *,
    exclude: Optional[str] = None,
    embedding_table_id: str = "all_entities",
    embedding_type_id: Optional[str] = None,
    pca_components: Optional[int] = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Download Wikipedia embeddings by type.

    More details on the embeddings can be found on
    https://soda-inria.github.io/ken_embeddings/.

    Parameters
    ----------
    types : str, optional
        Substring pattern that filters the types of entities.
        Will keep all entity types containing the substring.
        Write in lowercase. If `None`, all types will be passed.
    exclude : str, optional
        Type of embeddings to exclude from the types search.
    embedding_table_id : str, default='all_entities'
        Table of embedded entities from which to extract the embeddings.
        Get the supported tables with :func:`get_ken_table_aliases`.
        It is also possible to pass a custom figshare ID.
    embedding_type_id : str, optional
        Figshare ID of the file containing the type of embeddings.
        Get the supported tables with :func:`get_ken_types`.
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
    :func:`get_ken_table_aliases`
        Get the supported aliases of embedded entities tables.
    :func:`get_ken_types`
        Helper function to search for entity types.
    :func:`skrub.fuzzy_join` :
        Join two tables (dataframes) based on approximate column matching.
    :class:`skrub.FeatureAugmenter` :
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
    pyarrow installed to run correctly.

    The `types` parameter is there to filter the types by the input string
    pattern.
    In case the input is "music", all types with this string will be included
    (e.g. "wikicat_musician_from_france", "wikicat_music_label" etc.).
    Going directly for the exact type name (e.g. "wikicat_rock_music_bands")
    is possible but may not be complete (as some relevant bands may be
    in other similar types).
    For searching the types, the :func:`~skrub.datasets.get_ken_types`
    function can be used.

    """
    if embedding_table_id in get_ken_table_aliases():
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
    if types is not None:
        emb_type = emb_type[emb_type["Type"].str.contains(types)]
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
            emb_final.append(emb_extracts)
    emb_df = pd.concat(emb_final)
    return emb_df
