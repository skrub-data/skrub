from dirty_cat.datasets import get_ken_embeddings


def test_small_ken_embeddings():
    """
    Test if small sized embeddings were fetched correctly
    """
    emb = get_ken_embeddings(
        types="game_designers",
        embedding_table_id="games",
    )
    assert emb.shape[1] == 202

    # With custom figshare ID's:
    emb2 = get_ken_embeddings(
        types="game_designers",
        embedding_table_id="39254360",
        embedding_type_id="39266678",
        pca_components=5,
    )
    assert emb2.shape[1] == 7


def test_big_ken_embeddings():
    """
    Test if bigger sized embeddings were fetched correctly
    """
    # With custom figshare ID's:
    emb3 = get_ken_embeddings(
        types="rock",
        exclude="metal",
        embedding_table_id="39149066",
        embedding_type_id="39266300",
    )
    assert emb3.shape[1] == 202

    emb4 = get_ken_embeddings(
        types="pop",
        exclude="jazz",
        embedding_table_id="albums",
        pca_components=10,
    )
    assert emb4.shape[1] == 12
