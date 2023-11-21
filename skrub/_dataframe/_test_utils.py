def is_namespace_pandas(px):
    return px.__name__ == "pandas"


def is_namespace_polars(px):
    return px.__name__ == "polars"
