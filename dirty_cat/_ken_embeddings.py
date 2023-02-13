"""
Get the Wikipedia embeddings for feature augmentation.
"""
import pandas as pd
from sklearn.decomposition import PCA

from dirty_cat.datasets import fetch_embeddings


def get_ken_embeddings(
    types,
    exclude=None,
    pca_components=None,
    emb_full_id="39142985",
    emb_type_id="39143012",
    suffix="",
):
    """Extract and clean Wikipedia embeddings by type.

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
    suffix : str
        Suffix to add to the column names of the embeddings.

    Returns
    -------
    embeddings: class:`~pandas.DataFrame`
        The embeddings of entities of the specified type from Wikipedia.

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
    Relational data embeddings for feature enrichment with background information.
    https://doi.org/10.1007/s10994-022-06277-7
    """
    # Get all embeddings:
    # TODO: fetch embeddings from figshare in fetch.dataset
    # so as not to download them every time
    emb_full = fetch_embeddings(emb_full_id)
    emb_type = fetch_embeddings(emb_type_id)
    emb_full = pd.read_parquet(emb_full["path"])
    emb_type = pd.read_parquet(emb_type["path"])
    # All in lower case for easier matching
    emb_type["Type"] = emb_type["Type"].str.lower()
    if pca_components is not None:
        pca_i = PCA(n_components=pca_components, random_state=0)
        emb_columns = []
        for j in range(pca_components):
            name = "X" + str(j) + suffix
            emb_columns.append(name)
        pca_embeddings = pca_i.fit_transform(emb_full.drop(columns=["Entity"]))
        pca_embeddings = pd.DataFrame(pca_embeddings, columns=emb_columns)
        emb_pca = pd.concat([pca_embeddings, emb_full["Entity"]], axis=1)
        del pca_embeddings
        del emb_full
    else:
        emb_pca = emb_full
        del emb_full
    emb_pca = pd.merge(emb_pca, emb_type, on="Entity")
    del emb_type
    emb_extracts = emb_pca[emb_pca["Type"].str.contains(types)].drop_duplicates(
        subset=["Entity"]
    )
    emb_extracts["Entity"] = (
        emb_extracts["Entity"]
        .str.replace("<", "")
        .str.replace(">", "")
        .str.replace("_", " ")
    )
    emb_extracts.reset_index(drop=True, inplace=True)
    if exclude is not None:
        emb_extracts = emb_extracts[~emb_extracts["Type"].str.contains(exclude)]
    return emb_extracts
