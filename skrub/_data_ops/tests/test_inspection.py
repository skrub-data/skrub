import builtins
import functools
import linecache
import re
import sys
import traceback
import types
import webbrowser
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold

import skrub
from skrub import datasets
from skrub._data_ops import _inspection, _utils


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_output_dir(tmp_path):
    e = skrub.X()
    assert e.skb.full_report(open=False)["report_path"].is_relative_to(
        datasets.get_data_dir()
    )
    out = tmp_path / "report"
    assert (
        e.skb.full_report(open=False, output_dir=out)["report_path"]
        == out / "index.html"
    )
    with pytest.raises(FileExistsError, match=".*Set 'overwrite=True'"):
        e.skb.full_report(open=False, output_dir=out)

    assert (
        e.skb.full_report(open=False, output_dir=out, overwrite=True)["report_path"]
        == out / "index.html"
    )


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_full_report():
    # smoke test for the full report
    # TODO we should have a private function that returns the JSON data so we
    #      can check the content before rendering with jinja
    # however that requires first settling on the content of the report etc.
    e = -(
        (skrub.var("a", 12345) + 1).skb.set_name("b").skb.set_description("this is b")
        / skrub.var("c", 1)
    )
    report = e.skb.full_report(open=False)
    assert report["error"] is None
    assert report["result"] == -12346.0
    assert "-12346.0" in (report["report_path"].parent / "node_4.html").read_text(
        "utf-8"
    )
    report = e.skb.full_report({"a": 12345, "c": 0}, open=False)
    assert isinstance(report["error"], ZeroDivisionError)
    assert report["result"] is None
    out = report["report_path"].parent
    text = (out / "node_1.html").read_text("utf-8")
    assert "12346" in text and "this is b" in text
    assert "ZeroDivisionError" in (out / "node_3.html").read_text("utf-8")
    assert "This step did not run" in (out / "node_4.html").read_text("utf-8")


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_full_report_title():
    # TODO we should have a private function that returns the JSON data so we
    #      can check the content before rendering with jinja
    # however that requires first settling on the content of the report etc.
    data_op = skrub.var("a", 1)
    title = "small data ops"
    report = data_op.skb.full_report(open=False, title=title)
    assert title in report["report_path"].read_text("utf-8")


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_full_report_no_eval():
    data_op = skrub.var("a", 1) / skrub.var("b", 2)
    report = data_op.skb.full_report(open=False, eval=False)
    assert report["result"] is None
    assert report["error"] is None
    with pytest.raises(TypeError, match="must be None"):
        data_op.skb.full_report({"a": 1, "b": 0}, open=False, eval=False)


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_preview_subsample():
    X = datasets.fetch_employee_salaries().X
    preview = skrub.X(X).skb.subsample(n=3)._repr_html_()
    assert "subsample" in preview


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_full_report_failed_apply():
    # Somewhat contrived example for the corner case where an Apply does not
    # have an easily identifiable estimator.
    orders = skrub.datasets.toy_orders()
    e = (
        skrub.X()
        .skb.apply(SelectKBest())  # error! missing y
        .skb.apply(
            # never reached so there is no fitted estimator
            skrub.choose_from([DummyClassifier(), DummyClassifier()], name="classif"),
            y=skrub.y(),
        )
    )
    report = e.skb.full_report({"X": orders.X, "y": orders.y}, open=False)
    assert report["error"] is not None


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_full_report_dataop_estimator(tmp_path):
    # The estimator of an Apply can itself be a DataOp (the function/estimator
    # to apply is computed dynamically). Here the estimator's variable is not
    # provided so the node is not evaluated and there is no fitted
    # `estimator_`; the report must fall back on the DataOp.
    e = skrub.X().skb.apply(skrub.var("est"))
    report = e.skb.full_report(
        {"X": pd.DataFrame({"a": [1, 2]})},
        output_dir=tmp_path / "report",
        open=False,
    )
    assert report["error"] is not None
    assert report["result"] is None


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
@pytest.mark.parametrize(
    "estimator",
    [
        skrub.choose_from([DummyClassifier(), "passthrough"], name="est"),
        skrub.optional(DummyClassifier(), name="est"),
        "passthrough",
        None,
    ],
)
def test_unfitted_apply_no_doc_or_source(tmp_path, estimator):
    # When the Apply node is not evaluated (here because a value for its input
    # is not provided), there is no fitted `estimator_` and the report falls
    # back on the `estimator` passed to .skb.apply(). For choices,
    # "passthrough" and None we must not show the doc or source of the Choice,
    # str or NoneType class.
    report_dir = tmp_path / "report"
    e = skrub.var("a").skb.apply(estimator)
    report = e.skb.full_report(
        {},  # 'a' is missing from env so the apply does not get evaluated
        output_dir=report_dir,
        open=False,
    )
    assert report["error"] is not None
    text = (report_dir / "node_1.html").read_text("utf-8")
    assert "source code" not in text
    assert "docstring:" not in text


class _Doubler:
    """
    This is the docstring of _Doubler
    """

    # Does not inherit from BaseEstimator so it has no _repr_html_: the report
    # must still show its docstring and source.

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return X * 2

    def transform(self, X):
        return X * 2


def _times_two(x):
    """
    This is the docstring of _times_two
    """
    return x * 2


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
@pytest.mark.parametrize("no_wrap", [False, True])
def test_estimator_doc_and_source(tmp_path, no_wrap):
    report_dir = tmp_path / "report"
    df = pd.DataFrame({"a": [1, 2, 3]})

    # no_wrap: even if wrapped in ApplyToCols, the docstring of the wrapped
    # transformer should be shown (not that of ApplyToCols)
    skrub.X(df).skb.apply(_Doubler(), no_wrap=no_wrap).skb.full_report(
        output_dir=report_dir, open=False
    )
    assert "This is the docstring of _Doubler" in (
        report_dir / "node_1.html"
    ).read_text("utf-8")
    assert "X * 2" in next((report_dir / "python").glob("*.html")).read_text("utf-8")


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
@pytest.mark.parametrize("no_wrap", [False, True])
def test_fitted_passthrough_no_doc_or_source(tmp_path, no_wrap):
    # "passthrough" is replaced by a PassThrough, wrapped in ApplyToCols or not
    report_dir = tmp_path / "report"
    skrub.X(pd.DataFrame({"a": [1]})).skb.apply(
        "passthrough", no_wrap=no_wrap
    ).skb.full_report(output_dir=report_dir, open=False)
    text = (report_dir / "node_1.html").read_text("utf-8")
    assert "source code" not in text
    assert "docstring:" not in text


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_call_doc_and_source(tmp_path):
    report_dir = tmp_path / "report"
    skrub.var("a").skb.apply_func(_times_two).skb.full_report(
        {"a": 3}, output_dir=report_dir, open=False
    )
    assert "This is the docstring of _times_two" in (
        report_dir / "node_1.html"
    ).read_text("utf-8")
    assert "x * 2" in next((report_dir / "python").glob("*.html")).read_text("utf-8")

    report_dir = tmp_path / "report_lambda"
    skrub.var("a").skb.apply_func(lambda x: x).skb.full_report(
        {"a": 3}, output_dir=report_dir, open=False
    )
    assert "docstring:" not in (report_dir / "node_1.html").read_text("utf-8")


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_source_link_target_exists(tmp_path):
    # Check that the link to the source file is correct: we find the link in
    # the node page and verify the file exists.
    report_dir = tmp_path / "report"
    skrub.var("a").skb.apply_func(_times_two).skb.full_report(
        {"a": 3}, output_dir=report_dir, open=False
    )
    text = (report_dir / "node_1.html").read_text("utf-8")
    match = re.search(r'href="(python/[0-9a-f]+\.html)#L\d+"', text)
    assert (report_dir / match.group(1)).is_file()


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_get_source_url_mismatch(tmp_path):
    # no link if the source file does not contain the definition of the object
    assert _inspection._get_source_url(_times_two, tmp_path) is not None
    other = tmp_path / "other.py"
    other.write_text("\n" * 200)
    code = _times_two.__code__.replace(co_filename=str(other))
    f = types.FunctionType(code, _times_two.__globals__)
    assert _inspection._get_source_url(f, tmp_path) is None

    # classes and lambdas
    assert _inspection._get_source_url(_Doubler, tmp_path) is not None
    assert _inspection._get_source_url(lambda x: x, tmp_path) is not None

    # wrapped function: the link must point to the wrapped function's file
    @functools.wraps(_times_two)
    def wrapper(x):
        return _times_two(x)

    url = _inspection._get_source_url(wrapper, tmp_path)
    assert url == _inspection._get_source_url(_times_two, tmp_path)


def test_get_doc_partial():
    # for partials we show the docstring of the wrapped function
    expected = _inspection._get_doc(_times_two)
    assert _inspection._get_doc(functools.partial(_times_two)) == expected
    nested = functools.partial(functools.partial(_times_two))
    assert _inspection._get_doc(nested) == expected


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_stack_info_source_mismatch(tmp_path):
    # e.g. the DataOp was loaded from a cloudpickle and the lines in linecache
    # are for the wrong file (the one loading the pickle, not the one where the
    # function was defined). In this case we should get no link
    same = tmp_path / "same.py"
    same.write_text("import skrub\nx = skrub.var('a')\n")
    other = tmp_path / "other.py"
    other.write_text("import skrub\ny = 0\n")
    out = tmp_path / "report"
    out.mkdir()
    for path, has_link in [(same, True), (other, False)]:
        frame = (str(path), 2, "<module>", "x = skrub.var('a')")
        info = _inspection._get_stack_info([frame], out)
        assert (info[0]["url"] is not None) == has_link


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
@pytest.mark.parametrize("func_wrapper", ["dataop", "choice"])
def test_called_func_is_dataop_or_choice(tmp_path, func_wrapper):
    # .skb.apply_func() accepts a DataOp as func (the function to apply is
    # itself computed dynamically); no doc or source is available for it.
    report_dir = tmp_path / "report"
    func = (
        skrub.var("my_func", _times_two, becomes_default=True)
        if func_wrapper == "dataop"
        else skrub.choose_from([_times_two, _times_two], name="my_func")
    )
    skrub.var("a").skb.apply_func(func).skb.full_report(
        {"a": 3}, output_dir=report_dir, open=False
    )
    node = 2 if func_wrapper == "dataop" else 1
    text = (report_dir / f"node_{node}.html").read_text("utf-8")
    assert "Function applied in this step:" in text
    assert "my_func" in text
    assert "source code" not in text
    assert "docstring:" not in text


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_call_func_no_source(tmp_path):
    # builtins have a docstring but no retrievable source code.
    report_dir = tmp_path / "report"
    skrub.var("a").skb.apply_func(len).skb.full_report(
        {"a": [1, 2]}, output_dir=report_dir, open=False
    )
    text = (report_dir / "node_1.html").read_text("utf-8")
    assert "docstring:" in text
    assert "source code" not in text


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_call_func_from_linecache(tmp_path, monkeypatch):
    # functions defined in a Jupyter-style cell have no real
    # source file but their source can be found through linecache.
    report_dir = tmp_path / "report"
    filename = "<test-cell>"
    src = "def f(x):\n    '''doc for f'''\n    return x\n"
    namespace = {}
    exec(compile(src, filename, "exec"), namespace)
    monkeypatch.setitem(
        linecache.cache,
        filename,
        (len(src), None, src.splitlines(keepends=True), filename),
    )
    skrub.var("a").skb.apply_func(namespace["f"]).skb.full_report(
        {"a": 3}, output_dir=report_dir, open=False
    )
    source = next((report_dir / "python").glob("*.html")).read_text("utf-8")
    assert "def f(x):" in source
    assert "test-cell" in source


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_report_no_creation_stack(monkeypatch):
    monkeypatch.setattr(
        traceback, "extract_stack", Mock(side_effect=Exception("error"))
    )
    e = skrub.var("a") + 1
    monkeypatch.undo()
    report = e.skb.full_report({"a": 1}, open=False)
    text = (report["report_path"].parent / "node_1.html").read_text("utf-8")
    assert '<code class="node-creation-stack"></code>' in text


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_report_fit_mode():
    # non-regression: in fit mode the individual node pages used to show the
    # dataop itself as the output instead of the result of fit_transform for
    # intermediate nodes.
    learner = (
        skrub.var("a")
        .skb.apply_func(lambda x: x[::-1])
        .skb.apply_func(lambda x: x.upper())
        .skb.make_learner()
    )
    report = learner.report(environment={"a": "hello"}, mode="fit", open=False)
    report_dir = Path(report["report_path"]).parent
    node_1_report = (report_dir / "node_1.html").read_text("utf-8")
    # the page does display the actual output rather than the repr of the node
    # ('hello'[::-1] = 'olleh')
    assert "olleh" in node_1_report


def test_report_score_mode_with_scoring():
    learner = (
        skrub.X()
        .skb.apply(DummyClassifier(), y=skrub.y())
        .skb.with_scoring("accuracy")
        .skb.make_learner()
    )
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Creating the report for 'score' mode when .skb.with_scoring() "
            "has been used is not implemented yet."
        ),
    ):
        learner.report(environment={}, mode="score")


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_full_report_open(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(webbrowser, "open", mock)
    skrub.as_data_op(0).skb.full_report()
    mock.assert_called_once()


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_draw_graph():
    data_op = skrub.as_data_op(0)
    g = data_op.skb.draw_graph()
    assert repr(g) == "<GraphDrawing: use .open() to display>"
    assert b"<svg" in g.svg
    uuid = str(data_op.skb.id).encode("utf-8")
    assert uuid not in g.svg
    assert uuid in data_op.skb.draw_graph(show_ids=True).svg
    assert "<svg" in g._repr_html_()
    assert g.png.startswith(b"\x89PNG")
    assert g._repr_png_().startswith(b"\x89PNG")


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_draw_graph_split_x():
    # mark_as_X with a splitter creates a SplitX node which is already labelled
    # 'X' in its repr; _node_kwargs must not prefix it with another 'X:'.
    x = skrub.var("a").skb.mark_as_X(cv=KFold())
    assert _inspection._node_kwargs(x)["label"] == "X"
    svg = x.skb.draw_graph().svg.decode("utf-8")
    assert "X" in svg
    assert "X:\u2002X" not in svg


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_svg_anchor_google_colab(monkeypatch):
    """non-regression test for #1589"""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    svg = skrub.as_data_op(0).skb.set_description("SOME TEXT").skb.draw_graph().svg
    assert re.search(rb'<a target="_blank" xlink:title=".*SOME TEXT', svg)


def test_no_pydot(monkeypatch):
    monkeypatch.delitem(sys.modules, "pydot", raising=False)
    builtin_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "pydot":
            raise ImportError(name)
        return builtin_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(RuntimeError, match="please install Pydot and Graphviz"):
        skrub.as_data_op(0).skb.draw_graph()


def test_no_graphviz(monkeypatch):
    pydot = pytest.importorskip("pydot")
    monkeypatch.setattr(pydot.Dot, "create_svg", Mock(side_effect=Exception()))
    with pytest.raises(RuntimeError, match="please install Pydot and Graphviz"):
        skrub.as_data_op(0).skb.draw_graph()


def test_repr_html_no_graphviz(monkeypatch):
    monkeypatch.delitem(sys.modules, "pydot", raising=False)
    builtin_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "pydot":
            raise ImportError(name)
        return builtin_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    # Without a preview value (and without a graph) and with a preview value,
    # the HTML representation falls back on the graphviz error message.
    assert "please install Pydot and Graphviz" in skrub.var("a")._repr_html_()
    assert "please install Pydot and Graphviz" in skrub.var("a", 0)._repr_html_()


@pytest.mark.skipif(not _utils.has_graphviz(), reason="report requires graphviz")
def test_draw_graph_open(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(_inspection, "open_in_browser", mock)
    skrub.as_data_op(0).skb.draw_graph().open()
    mock.assert_called_once()


def test_describe_param_grid():
    """
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.preprocessing import StandardScaler, RobustScaler

    >>> import skrub

    >>> X = skrub.X()
    >>> y = skrub.y()

    >>> imputed = X.skb.apply(skrub.optional(SimpleImputer(), name="impute"))
    >>> dim_reduction = skrub.choose_from(
    ...     {
    ...         "PCA": PCA(),
    ...         "SelectKBest": SelectKBest(),
    ...     },
    ...     name="dim_reduction",
    ... )
    >>> selected = imputed.skb.apply(dim_reduction)
    >>> use_scaling = skrub.choose_bool(name="scaling")
    >>> scaling_kind = skrub.choose_from(["robust", "standard"], name="scaling_kind")
    >>> scaler = scaling_kind.match(
    ...     {
    ...         "robust": RobustScaler(
    ...             **skrub.choose_bool(name="robust_scaler__with_centering")
    ...         ),
    ...         "standard": StandardScaler(),
    ...     }
    ... )
    >>> scaled = selected.skb.apply(use_scaling.if_else(scaler, None))
    >>> classifier = skrub.choose_from(
    ...     {
    ...         "logreg": LogisticRegression(
    ...             **skrub.choose_float(0.001, 100, log=True, name="C")
    ...         ),
    ...         "rf": RandomForestClassifier(
    ...             n_estimators=skrub.choose_int(20, 400, name="N 🌴")
    ...         ),
    ...     },
    ...     name="classifier",
    ... )
    >>> pred = scaled.skb.apply(classifier, y=y)

    those need to be split into separate subgrids:

    - scaling or not because the nested choice scaling kind is only used if
      scaling is true
    - scaling kind because the robust scaler has a nested choice with_centering
    - the classifier because the random forest and logistic regression have
      nested choices (their hyperparameters)

    so we end up with 2 (rf or logreg) x 3 (no scaling, robust, standard) = 6
    subgrids.

    >>> print(pred.skb.describe_param_grid())
    - impute: [SimpleImputer(), None]
      dim_reduction: ['PCA', 'SelectKBest']
      scaling: True
      scaling_kind: 'robust'
      robust_scaler__with_centering: [True, False]
      classifier: 'logreg'
      C: choose_float(0.001, 100, log=True, name='C')
    - impute: [SimpleImputer(), None]
      dim_reduction: ['PCA', 'SelectKBest']
      scaling: True
      scaling_kind: 'robust'
      robust_scaler__with_centering: [True, False]
      classifier: 'rf'
      N 🌴: choose_int(20, 400, name='N 🌴')
    - impute: [SimpleImputer(), None]
      dim_reduction: ['PCA', 'SelectKBest']
      scaling: True
      scaling_kind: 'standard'
      classifier: 'logreg'
      C: choose_float(0.001, 100, log=True, name='C')
    - impute: [SimpleImputer(), None]
      dim_reduction: ['PCA', 'SelectKBest']
      scaling: True
      scaling_kind: 'standard'
      classifier: 'rf'
      N 🌴: choose_int(20, 400, name='N 🌴')
    - impute: [SimpleImputer(), None]
      dim_reduction: ['PCA', 'SelectKBest']
      scaling: False
      classifier: 'logreg'
      C: choose_float(0.001, 100, log=True, name='C')
    - impute: [SimpleImputer(), None]
      dim_reduction: ['PCA', 'SelectKBest']
      scaling: False
      classifier: 'rf'
      N 🌴: choose_int(20, 400, name='N 🌴')
    """


def test_describe_params():
    c1 = skrub.choose_float(0.0, 1.0)
    c2 = skrub.choose_from((5.5, c1), name="c2")
    c3 = skrub.choose_bool()
    c4 = skrub.choose_from({"2": c2, "1": c1, "3": c3})
    c5 = skrub.choose_int(10, 20, default=11, name="c5")
    c6 = skrub.choose_from(["a", "b"])
    c7 = skrub.choose_float(100.0, 200.0, default=110.5)
    e = c6.match({"a": [c4, c2.as_data_op() + c7], "b": c5}).as_data_op()
    print(e.skb.describe_defaults())
    expected = {
        "choose_from(['a', 'b'])": "a",
        "choose_from({'2': …, '1': …, '3': …})": "2",
        "c2": 5.5,
        "choose_float(100.0, 200.0, default=110.5)": 110.5,
    }

    assert e.skb.describe_defaults() == expected
    assert e.skb.make_learner().describe_params() == expected
    assert skrub.X().skb.describe_defaults() == {}
