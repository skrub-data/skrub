def asdfapi(obj):
    if hasattr(obj, "__dataframe_namespace__"):
        return obj
    if hasattr(obj, "__column_namespace__"):
        return obj
    if hasattr(obj, "__dataframe_consortium_standard__"):
        return obj.__dataframe_consortium_standard__()
    if hasattr(obj, "__column_consortium_standard__"):
        return obj.__column_consortium_standard__()
    raise TypeError(
        f"{obj} cannot be converted to DataFrame Consortium Standard object."
    )


def asnative(obj):
    if hasattr(obj, "__dataframe_namespace__"):
        return obj.dataframe
    if hasattr(obj, "__column_namespace__"):
        return obj.column
    if hasattr(obj, "__scalar_namespace__"):
        return obj.scalar
    return obj


def dfapi_ns(obj):
    obj = asdfapi(obj)
    for attr in [
        "__dataframe_namespace__",
        "__column_namespace__",
        "__scalar_namespace__",
    ]:
        if hasattr(obj, attr):
            return getattr(obj, attr)()
    raise TypeError(f"{obj} is not a Dataframe Consortium Standard object.")
