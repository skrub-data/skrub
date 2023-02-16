"""
Get the Wikipedia embeddings for feature augmentation.
"""
import pandas as pd
from sklearn.decomposition import PCA

from dirty_cat.datasets import fetch_figshare


def get_ken_embeddings(
    types=None,
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
    types : str, optional, default=None
        Substring pattern that filters the types of entities.
        Will keep all entity types containing the substring.
        Write in lowercase. If None, all types will be passed.
    exclude : str, optional, default=None
        Type of embeddings to exclude from the types search.
    embedding_table_id : {"all_entities", "albums", "companies", "movies", "games", "school"} or str, optional, default='all_entities' # noqa
        Table of embedded entities from which to extract the embeddings.
        See correspondence table
        (https://github.com/dirty-cat/datasets/blob/master/data/ken_correspondence.csv)
        for the figshare ID's of the tables.
        It is also possible to introduce a custom figshare ID.
    embedding_type_id : str, optional, default=None
        Figshare ID of the file containing the type of embeddings. Ignored
        unless a custom `embedding_table_id` is provided.
    pca_components : int, optional, default=None
        Size of the dimensional space on which the embeddings will be projected
        by a principal component analysis.
        If None, the default dimension (200) of the embeddins will be kept.
    suffix : str, optional, default=""
        Suffix to add to the column names of the embeddings.

    Returns
    -------
    embeddings: class:`~pandas.DataFrame`
        The embeddings of entities and the specified type from Wikipedia.

    See Also
    --------
    :class:`~dirty_cat.fuzzy_join` :
        Join two tables (dataframes) based on approximate column matching.
    :class:`~dirty_cat.FeatureAugmenter` :
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
    For the full list of accepted types, see the `entity_detailed_types`
    table on https://soda-inria.github.io/ken_embeddings/.

    """
    if embedding_table_id in [
        "all_entities",
        "albums",
        "companies",
        "movies",
        "games",
        "school",
    ]:
        correspondence = pd.read_csv(
            "https://raw.githubusercontent.com/dirty-cat/datasets/master/data/ken_correspondence.csv"  # noqa
        )
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
    emb_df = pd.DataFrame()
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
