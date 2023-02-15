from dirty_cat.datasets import get_ken_embeddings


def test_small_ken_embeddings():
    """
    Test if small sized embeddings were fetched correctly
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


def test_big_ken_embeddings():
    """
    Test if big sized embeddings were fetched correctly
    """
    emb3 = get_ken_embeddings(
        types="game_publish",
        exclude="company",
        emb_id="39254360",
    )
    assert emb3.shape[1] == 202

    emb4 = get_ken_embeddings(
        types="game_publish",
        exclude="company",
        emb_id="39254360",
        pca_components=10,
    )
    assert emb4.shape[1] == 12
