"""Use a DataOp computation graph as the thumbnail of gallery examples."""

from pathlib import Path


def scrape_data_op_graph(block, block_vars, gallery_conf):
    """Write the last DataOp graph displayed in the example as its thumbnail.

    The scraper does not add any reST to the example: the graph is already
    displayed in the example body by the ``DataOp._repr_html_`` of the last
    expression. It only writes an SVG file where Sphinx-Gallery looks for the
    example's first image, so that the graph is used as the gallery thumbnail.

    The SVG is deliberately written without consuming an entry from
    ``image_path_iterator``. If the example also produces matplotlib figures,
    those are saved to the same path with a ``.png`` extension, and
    Sphinx-Gallery looks for ``.png`` before ``.svg``: matplotlib figures
    therefore keep priority over the graph. Since each DataOp output overwrites
    the same SVG, the last graph shown in the example is the one kept.
    """
    data_op = block_vars.get("example_globals", {}).get("___")
    if data_op is None:
        return ""
    try:
        import skrub
    except ImportError:
        return ""
    if not isinstance(data_op, skrub.DataOp):
        return ""
    image_path_iterator = block_vars["image_path_iterator"]
    if len(image_path_iterator):
        # Another scraper (typically matplotlib) already produced an image for
        # this example, and such images take priority over the graph.
        return ""
    try:
        svg = data_op.skb.draw_graph().svg
    except Exception:
        # Graphviz is not installed (or rendering failed): degrade gracefully
        # and let Sphinx-Gallery fall back to the default thumbnail.
        return ""
    image_path = Path(str(image_path_iterator.image_path).format(1)).with_suffix(".svg")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(svg)
    return ""
