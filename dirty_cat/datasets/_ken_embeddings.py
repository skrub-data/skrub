"""
Get the Wikipedia embeddings for feature augmentation.
"""
import pandas as pd
from sklearn.decomposition import PCA

from dirty_cat.datasets import fetch_figshare


def get_ken_embeddings(
    types,
    exclude=None,
    pca_components=None,
    emb_id="39254360",
    emb_type_id="39143012",
    suffix="",
):
    """Extract Wikipedia embeddings by type.

    More details on the embeddings can be found on
    https://soda-inria.github.io/ken_embeddings/.

    Parameters
    ----------
    types : str
        Types of embeddings to include. Write in lowercase.
    exclude : str, default=None
        Type of embeddings to exclude from the types search.
    pca_components : int, default=None
        Size of the dimensional space on which the embeddings will be projected
        by a principal component analysis.
        If None, the default dimension (200) of the embeddins will be kept.
    emb_id : str
        Figshare ID of the file containing all embeddings.
    emb_type_id : str
        Figshare ID of the file containing the type of embeddings.
    suffix : str
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
        Transformer to enrich a given table via one or more fuzzy joins to external
        resources.

    References
    ----------
    For more details, see Cvetkov-Iliev, A., Allauzen, A. & Varoquaux, G.:
    `Relational data embeddings for feature enrichment
    with background information. <https://doi.org/10.1007/s10994-022-06277-7>`_

    Notes
    -----
    The files are read and returned in parquet format, this function needs
    pyarrow installed to run correctly.

    """
    # Get all embeddings:
    emb_type = fetch_figshare(emb_type_id)
    emb_type = pd.read_parquet(emb_type["path"])
    # All in lower case for easier matching
    emb_type["Type"] = emb_type["Type"].str.lower()
    emb_type = emb_type[emb_type["Type"].str.contains(types)]
    emb_type.drop_duplicates(subset=["Entity"], inplace=True)
    if exclude is not None:
        emb_type = emb_type[~emb_type["Type"].str.contains(exclude)]
    emb_final = []
    emb_df = pd.DataFrame()
    emb_full = fetch_figshare(emb_id)
    for path in emb_full["path"]:
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
            emb_df = pd.concat(emb_final)
        else:
            emb_final.append(emb_extracts)
            emb_df = pd.concat(emb_final)
    return emb_df
