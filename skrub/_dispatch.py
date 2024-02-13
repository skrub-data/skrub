from functools import singledispatch


def _load_dataframe_module_info(name):
    # if the module is not installed, import errors are propagated
    if name == "pandas":
        import pandas

        return {
            "module": pandas,
            "types": {
                "DataFrame": [pandas.DataFrame],
                "Column": [pandas.Series],
            },
        }
    if name == "polars":
        import polars

        return {
            "module": polars,
            "types": {
                "DataFrame": [polars.DataFrame, polars.LazyFrame],
                "LazyFrame": [polars.LazyFrame],
                "EagerFrame": [polars.DataFrame],
                "Column": [polars.Series],
            },
        }
    raise KeyError(f"Unknown module: {name}")


def dispatch(function):
    dispatched = singledispatch(function)

    def specialize(module_name, generic_type_names=None):
        try:
            module_info = _load_dataframe_module_info(module_name)
        except (ImportError, KeyError):

            def decorator(specialized_impl):
                return specialized_impl

            return decorator

        if generic_type_names is None:
            generic_type_names = list(module_info["types"].keys())
        if isinstance(generic_type_names, str):
            generic_type_names = [generic_type_names]

        def decorator(specialized_impl):
            types_to_register = set()
            for type_name in generic_type_names:
                types_to_register.update(module_info["types"][type_name])
            for module_type in types_to_register:
                dispatched.register(module_type, specialized_impl)
            return specialized_impl

        return decorator

    dispatched.specialize = specialize
    return dispatched
