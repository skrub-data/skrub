import inspect
import re
from importlib import import_module
from typing import Optional

import pytest

numpydoc_validation = pytest.importorskip("numpydoc.validate")

FUNCTION_DOCSTRING_IGNORE_SET = {
    "dirty_cat.datetime_encoder.DatetimeEncoder",
    "dirty_cat.datetime_encoder.DatetimeEncoder.fit",
    "dirty_cat.datetime_encoder.DatetimeEncoder.fit_transform",
    "dirty_cat.datetime_encoder.DatetimeEncoder.get_feature_names",
    "dirty_cat.datetime_encoder.DatetimeEncoder.get_feature_names_out",
    "dirty_cat.datetime_encoder.DatetimeEncoder.get_params",
    "dirty_cat.datetime_encoder.DatetimeEncoder.set_params",
    "dirty_cat.datetime_encoder.DatetimeEncoder.transform",
    "dirty_cat.gap_encoder.GapEncoder",
    "dirty_cat.gap_encoder.GapEncoder.fit",
    "dirty_cat.gap_encoder.GapEncoder.fit_transform",
    "dirty_cat.gap_encoder.GapEncoder.get_feature_names",
    "dirty_cat.gap_encoder.GapEncoder.get_feature_names_out",
    "dirty_cat.gap_encoder.GapEncoder.get_params",
    "dirty_cat.gap_encoder.GapEncoder.partial_fit",
    "dirty_cat.gap_encoder.GapEncoder.score",
    "dirty_cat.gap_encoder.GapEncoder.set_params",
    "dirty_cat.gap_encoder.GapEncoder.transform",
    "dirty_cat.minhash_encoder.MinHashEncoder",
    "dirty_cat.minhash_encoder.MinHashEncoder.fit",
    "dirty_cat.minhash_encoder.MinHashEncoder.fit_transform",
    "dirty_cat.minhash_encoder.MinHashEncoder.get_fast_hash",
    "dirty_cat.minhash_encoder.MinHashEncoder.get_params",
    "dirty_cat.minhash_encoder.MinHashEncoder.get_unique_ngrams",
    "dirty_cat.minhash_encoder.MinHashEncoder.minhash",
    "dirty_cat.minhash_encoder.MinHashEncoder.set_params",
    "dirty_cat.minhash_encoder.MinHashEncoder.transform",
    "dirty_cat.similarity_encoder.SimilarityEncoder",
    "dirty_cat.similarity_encoder.SimilarityEncoder.fit",
    "dirty_cat.similarity_encoder.SimilarityEncoder.fit_transform",
    "dirty_cat.similarity_encoder.SimilarityEncoder.get_feature_names",
    "dirty_cat.similarity_encoder.SimilarityEncoder.get_feature_names_out",
    "dirty_cat.similarity_encoder.SimilarityEncoder.get_most_frequent",
    "dirty_cat.similarity_encoder.SimilarityEncoder.get_params",
    "dirty_cat.similarity_encoder.SimilarityEncoder.infrequent_categories_",
    "dirty_cat.similarity_encoder.SimilarityEncoder.inverse_transform",
    "dirty_cat.similarity_encoder.SimilarityEncoder.set_params",
    "dirty_cat.similarity_encoder.SimilarityEncoder.transform",
    "dirty_cat.super_vectorizer.SuperVectorizer",
    "dirty_cat.super_vectorizer.SuperVectorizer.OptionalEstimator",
    "dirty_cat.super_vectorizer.SuperVectorizer.fit",
    "dirty_cat.super_vectorizer.SuperVectorizer.fit_transform",
    "dirty_cat.super_vectorizer.SuperVectorizer.get_feature_names",
    "dirty_cat.super_vectorizer.SuperVectorizer.get_feature_names_out",
    "dirty_cat.super_vectorizer.SuperVectorizer.get_params",
    "dirty_cat.super_vectorizer.SuperVectorizer.named_transformers_",
    "dirty_cat.super_vectorizer.SuperVectorizer.set_params",
    "dirty_cat.super_vectorizer.SuperVectorizer.transform",
    "dirty_cat.target_encoder.TargetEncoder",
    "dirty_cat.target_encoder.TargetEncoder.fit",
    "dirty_cat.target_encoder.TargetEncoder.fit_transform",
    "dirty_cat.target_encoder.TargetEncoder.get_params",
    "dirty_cat.target_encoder.TargetEncoder.set_params",
    "dirty_cat.target_encoder.TargetEncoder.transform",
}


def all_estimators():
    module = import_module("dirty_cat")
    classes = inspect.getmembers(module, inspect.isclass)
    classes = [(name, est_cls) for name, est_cls in classes if not name.startswith("_")]
    return sorted(classes, key=lambda x: x[0])


def get_all_methods():
    estimators = all_estimators()
    for name, Estimator in estimators:
        if name.startswith("_"):
            # skip private classes
            continue
        methods = []
        for name in dir(Estimator):
            if name.startswith("_"):
                continue
            method_obj = getattr(Estimator, name)
            if hasattr(method_obj, "__call__") or isinstance(method_obj, property):
                methods.append(name)
        methods.append(None)

        for method in sorted(methods, key=str):
            yield Estimator, method


def repr_errors(res, estimator=None, method: Optional[str] = None) -> str:
    """Pretty print original docstring and the obtained errors

    Parameters
    ----------
    res : dict
        result of numpydoc.validate.validate
    estimator : {estimator, None}
        estimator object or None
    method : str
        if estimator is not None, either the method name or None.

    Returns
    -------
    str
       String representation of the error.
    """
    if method is None:
        if hasattr(estimator, "__init__"):
            method = "__init__"
        elif estimator is None:
            raise ValueError("At least one of estimator, method should be provided")
        else:
            raise NotImplementedError

    if estimator is not None:
        obj = getattr(estimator, method)
        try:
            obj_signature = str(inspect.signature(obj))
        except TypeError:
            # In particular we can't parse the signature of properties
            obj_signature = (
                "\nParsing of the method signature failed, "
                "possibly because this is a property."
            )

        obj_name = estimator.__name__ + "." + method
    else:
        obj_signature = ""
        obj_name = method

    msg = "\n\n" + "\n\n".join(
        [
            str(res["file"]),
            obj_name + obj_signature,
            res["docstring"],
            "# Errors",
            "\n".join(
                " - {}: {}".format(code, message) for code, message in res["errors"]
            ),
        ]
    )
    return msg


def filter_errors(errors, method, Estimator=None):
    """
    Ignore some errors based on the method type.
    """
    for code, message in errors:
        # We ignore following error code,
        #  - RT02: The first line of the Returns section
        #    should contain only the type, ..
        #   (as we may need refer to the name of the returned
        #    object)
        #  - GL01: Docstring text (summary) should start in the line
        #    immediately after the opening quotes (not in the same line,
        #    or leaving a blank line in between)
        #  - GL02: If there's a blank line, it should be before the
        #    first line of the Returns section, not after (it allows to have
        #    short docstrings for properties).

        if code in ["RT02", "GL01", "GL02"]:
            continue

        # Following codes are only taken into account for the
        # top level class docstrings:
        #  - ES01: No extended summary found
        #  - SA01: See Also section not found
        #  - EX01: No examples section found

        if method is not None and code in ["EX01", "SA01", "ES01"]:
            continue

        yield code, message


@pytest.mark.parametrize("Estimator, method", get_all_methods())
def test_docstring(Estimator, method, request):
    base_import_path = Estimator.__module__
    import_path = [base_import_path, Estimator.__name__]
    if method is not None:
        import_path.append(method)

    import_path = ".".join(import_path)

    if import_path in FUNCTION_DOCSTRING_IGNORE_SET:
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )

    res = numpydoc_validation.validate(import_path)

    res["errors"] = list(filter_errors(res["errors"], method, Estimator=Estimator))

    if res["errors"]:
        msg = repr_errors(res, Estimator, method)

        raise ValueError(msg)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate docstrings with numpydoc.")
    parser.add_argument("import_path", help="Import path to validate")

    args = parser.parse_args()

    res = numpydoc_validation.validate(args.import_path)

    import_path_sections = args.import_path.split(".")

    if len(import_path_sections) >= 2 and re.match(
        r"(?:[A-Z][a-z]*)+", import_path_sections[-2]
    ):
        method = import_path_sections[-1]
    else:
        method = None

    if res["errors"]:
        msg = repr_errors(res, method=args.import_path)

        print(msg)
        sys.exit(1)
    else:
        print("All docstring checks passed for {}!".format(args.import_path))
