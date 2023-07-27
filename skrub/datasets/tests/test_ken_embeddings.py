import pytest

from skrub.datasets import (
    fetch_ken_embeddings,
    fetch_ken_table_aliases,
    fetch_ken_types,
)


def test_fetch_ken_table_aliases():
    """
    Test if the aliases of the tables are correctly fetched
    """
    aliases = fetch_ken_table_aliases()
    assert "all_entities" in aliases
    assert "games" in aliases
    assert "albums" in aliases


def test_fetch_ken_types():
    """
    Test if the types of entities are correctly fetched
    """
    pytest.importorskip("pyarrow")
    # Tests the full result returns alright
    types = fetch_ken_types()
    assert types.shape[0] == 114509
    # Tests the additive filter works
    types2 = fetch_ken_types(search="game")
    assert types2.shape[0] == 2540
    # Tests the negative filter works
    types3 = fetch_ken_types(search="game", exclude="card")
    assert types3.shape[0] == 2532


def test_small_ken_embeddings():
    """
    Test if small sized embeddings were fetched correctly
    """
    pytest.importorskip("pyarrow")
    emb = fetch_ken_embeddings(
        search_types="game_designers",
        embedding_table_id="games",
    )
    assert emb.shape[1] == 202

    # With custom figshare ID's:
    emb2 = fetch_ken_embeddings(
        search_types="game_designers",
        embedding_table_id="39254360",
        embedding_type_id="39266678",
        pca_components=5,
    )
    assert emb2.shape[1] == 7


def test_big_ken_embeddings():
    """
    Test if bigger sized embeddings were fetched correctly
    """
    pytest.importorskip("pyarrow")
    # With custom figshare ID's:
    emb3 = fetch_ken_embeddings(
        search_types="rock",
        exclude="metal",
        embedding_table_id="39149066",
        embedding_type_id="39266300",
    )
    assert emb3.shape[1] == 202

    emb4 = fetch_ken_embeddings(
        search_types="pop",
        exclude="jazz",
        embedding_table_id="albums",
        pca_components=10,
    )
    assert emb4.shape[1] == 12
