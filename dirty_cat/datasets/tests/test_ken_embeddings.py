from dirty_cat.datasets import get_ken_embeddings


def test_fuzzy_join_dtypes():
    """
    Test if the embeddings were fetched correctly
    """
    emb = get_ken_embeddings(
        types="game_designers", emb_id="39254360", emb_type_id="39266678"
    )
    assert emb.shape[1] == 202

    emb2 = get_ken_embeddings(
        types="game_designers",
        emb_id="39254360",
        emb_type_id="39266678",
        pca_components=5,
    )
    assert emb2.shape[1] == 7
