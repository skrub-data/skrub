from dirty_cat import get_ken_embeddings


def test_fuzzy_join_dtypes():
    """
    Test if the embeddings were dowloaded correctly
    """
    emb = get_ken_embeddings(types="school")
    assert emb.shape == (201,)

    emb2 = get_ken_embeddings(types="school", pca_components=5)
    assert emb2.shape == (201,)
