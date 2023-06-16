"""
Get the Wikipedia embeddings for feature augmentation.
"""

import pandas as pd
from sklearn.decomposition import PCA

from skrub.datasets import fetch_figshare

# Required for ignoring lines too long in the docstrings
# flake8: noqa: E501


_correspondence_table_url = (
    "https://raw.githubusercontent.com/dirty-cat/datasets"
    "/master/data/ken_correspondence.csv"
)


def fetch_ken_table_aliases() -> set[str]:
    """Get the supported aliases of embedded KEN entities tables.

    These aliases can be using in subsequent functions (see section *See Also*).

    Returns
    -------
    set of str
        The aliases of the embedded entities tables.

    See Also
    --------
    :func:`fetch_ken_types`
        Helper function to search for entity types.
    :func:`fetch_ken_embeddings`
        Download Wikipedia embeddings by type.

    Notes
    -----
    Requires an Internet connection to work.

    Examples
    --------
    Let's see what are the current KEN subtables available
    for download:
    >>> fetch_ken_table_aliases()
    {'games', 'companies', 'schools', 'albums', 'all_entities', 'movies'}
    """
    correspondence = pd.read_csv(_correspondence_table_url)
    return set(["all_entities"] + list(correspondence["table"].values))


def fetch_ken_types(
    search: str = None,
    *,
    exclude: str | None = None,
    embedding_table_id: str = "all_entities",
) -> pd.DataFrame:
    """Helper function to search for KEN entity types.

    The result can then be used with :func:`fetch_ken_embeddings`.

    Parameters
    ----------
    search : str, optional
        Substring pattern that filters the types of entities.
    exclude : str, optional
        Substring pattern to exclude from the search.
    embedding_table_id : str, default='all_entities'
        Table of embedded entities from which to extract the embeddings.
        Get the supported tables with :func:`fetch_ken_table_aliases`.
        It is NOT possible to pass a custom figshare ID.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        The types of entities containing the substring.

    See Also
    --------
    :func:`fetch_ken_embeddings`
        Download Wikipedia embeddings by type.

    References
    ----------
    For more details, see Cvetkov-Iliev, A., Allauzen, A. & Varoquaux, G.:
    `Relational data embeddings for feature enrichment
    with background information. <https://doi.org/10.1007/s10994-022-06277-7>`_

    Notes
    -----
    Best used in conjunction with :func:`fetch_ken_embeddings`.

    Examples
    --------
    To get all the existing KEN types of entities:

    >>> embedding_types = fetch_ken_types()
    >>> embedding_types.head()
                                Type
    0                 wikicat_italian_male_screenwriters
    1  wikicat_21st-century_roman_catholic_archbishop...
    2                 wikicat_2000s_romantic_drama_films
    3                  wikicat_music_festivals_in_france
    4        wikicat_20th-century_american_women_artists

    Let's search for all KEN types with the strings "dance" or "music":

    >>> embedding_filtered_types = fetch_ken_types(search="dance|music")
    >>> embedding_filtered_types.head()
                                    Type
    0                    wikicat_music_festivals_in_france
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
    search_types: str | None = None,
    *,
    exclude: str | None = None,
    embedding_table_id: str = "all_entities",
    embedding_type_id: str | None = None,
    pca_components: int | None = None,
    suffix: str = "",
) -> pd.DataFrame:
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
        Get the supported tables with :func:`fetch_ken_table_aliases`.
        It is also possible to pass a custom figshare ID.
    embedding_type_id : str, optional
        Figshare ID of the file containing the type of embeddings.
        Get the supported tables with :func:`fetch_ken_types`.
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
    :func:`fetch_ken_table_aliases`
        Get the supported aliases of embedded entities tables.
    :func:`fetch_ken_types`
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

    The `search_types` parameter is there to filter the types by the input string
    pattern.
    In case the input is "music", all types with this string will be included
    (e.g. "wikicat_musician_from_france", "wikicat_music_label" etc.).
    Going directly for the exact type name (e.g. "wikicat_rock_music_bands")
    is possible but may not be complete (as some relevant bands may be
    in other similar types).
    For exploring available types, the :func:`~skrub.datasets.fetch_ken_types`
    function can be used.

    Examples
    --------
    :func:`fetch_ken_embeddings` allows you to extract embeddings
    you are interested in. For instance, if we are interested in
    video games:

    >>> games_embedding = fetch_ken_embeddings(search_types="video_games")
    >>> games_embedding.head()
                    Entity                       Type                       ...      X198      X199
    0       The_Mysterious_Island  wikicat_novels_adapted_into_video_games  ... -0.072814 -0.156973
    1    Storyteller_(video_game)             wikicat_upcoming_video_games  ...  0.059816  0.021077
    2  Skull_&_Bones_(video_game)        wikicat_video_games_about_pirates  ... -0.141682  0.024204
    3               Ethan_Winters   wikicat_male_characters_in_video_games  ... -0.107913 -0.089531
    4                     Cruis'n               wikicat_racing_video_games  ... -0.260757  0.060700

    Extracts all embeddings with the "games" type.
    For the list of existing types see :func:`fetch_ken_types`.

    Some tables are available pre-filtered for us using the
    `embedding_table_id` parameter:

    >>> games_embedding_fast = fetch_ken_embeddings(embedding_table_id="games")
    >>> games_embedding_fast.head()
                     Entity                               Type                      ...      X198      X199
    0              R-Type_Delta                                 wikicat_irem_games  ... -0.125806  0.040006
    1  Just_Add_Water_(company)  wikicat_video_game_companies_of_the_united_kin...  ...  0.067210 -0.025676
    2                 Li_Xiayan          wikicat_asian_games_medalists_in_swimming  ... -0.104818  0.003485
    3             Vampire_Night                        wikicat_vampire_video_games  ... -0.118209 -0.145383
    4               Shatterhand                             wikicat_platform_games  ... -0.138462  0.197820

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
            emb_final.append(emb_extracts)
    emb_df = pd.concat(emb_final)
    emb_df["Entity"] = emb_df["Entity"].str[1:-1]
    emb_df["Type"] = emb_df["Type"].str[1:-1]
    return emb_df.reset_index(drop=True)
