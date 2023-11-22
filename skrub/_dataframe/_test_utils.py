def is_module_pandas(px):
    return px.__name__ == "pandas"


def is_module_polars(px):
    return px.__name__ == "polars"
