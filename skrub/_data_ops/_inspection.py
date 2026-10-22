import base64
import copy
import datetime
import functools
import hashlib
import html
import inspect
import io
import linecache
import numbers
import re
import shutil
import sys
import traceback
import uuid
import webbrowser
from pathlib import Path

import jinja2
import numpy as np
import pydot

from .. import _dataframe as sbd
from .. import datasets
from .._config import get_config
from .._reporting import TableReport
from .._reporting._serve import open_in_browser
from .._reporting._utils import strip_xml_declaration
from .._utils import PassThrough, Repr, format_duration, random_string, short_repr
from . import _utils
from ._choosing import BaseChoice, BaseNumericChoice, Choice
from ._data_ops import Apply, Call, DataOp, SplitX, Value, Var
from ._evaluation import choice_graph, clear_results, evaluate, graph, param_grid
from ._subsampling import uses_subsampling


def _get_jinja_env():
    templates_dir = (
        Path(__file__).resolve().parents[1]
        / "_reporting"
        / "_data"
        / "templates"
        / "data_ops"
    )
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_dir, encoding="UTF-8"),
        autoescape=True,
    )
    env.filters["format_duration"] = format_duration
    env.globals["uuid"] = str(uuid.uuid4())
    # Not loaded as a template because it contains sequences that jinja would
    # interpret. See wasm-graphviz/README.md
    env.globals["graphviz_wasm_script"] = (
        templates_dir / "wasm-graphviz" / "graphviz.js"
    ).read_text("utf-8")
    return env


def _get_template(template_name):
    return _get_jinja_env().get_template(template_name)


def _use_table_report_display():
    return get_config()["use_table_report_data_ops"]


def node_report(data_op, mode="preview", environment=None, **report_kwargs):
    result = evaluate(data_op, mode=mode, environment=environment)
    if sbd.is_column(result):
        # TODO say in page that it was a column not df
        # maybe this should be handled by tablereport? or we should have a
        # seriesreport with just 1 card?
        result_df = sbd.make_dataframe_like(result, [result])
        result_df = sbd.copy_index(result, result_df)
        result = result_df
    if sbd.is_dataframe(result) and _use_table_report_display():
        report_kwargs.setdefault("verbose", False)  # Hide the progress bar
        report = TableReport(result, **report_kwargs)
        report._set_minimal_mode()
        if uses_subsampling(data_op):
            report._display_subsample_hint()
    else:
        try:
            report = result._repr_html_()
        except Exception:
            res_repr = Repr()
            res_repr.maxstring = 1000
            res_repr.maxother = 10_000
            report = _get_template("simple-repr.html").render(
                {"object_repr": res_repr.repr(result)}
            )
    return report


def _get_output_dir(output_dir, overwrite):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    if output_dir is None:
        output_base_dir = datasets.get_data_dir() / "execution_reports"
        output_dir = output_base_dir / f"full_data_op_report_{now}_{random_string()}"
        _utils.prune_directory(output_base_dir)
    else:
        output_dir = Path(output_dir).expanduser().resolve()
        if output_dir.exists():
            if overwrite:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"The output directory already exists: {output_dir}. "
                    "Set 'overwrite=True' to allow replacing it."
                )
    output_dir.mkdir(parents=True)
    return output_dir


def _node_status(data_op_graph, mode, eval):
    status = {}
    for node_id, node in data_op_graph["nodes"].items():
        if not eval:
            status[node_id] = "global_no_eval"
        elif mode in node._skrub_impl.results:
            status[node_id] = "success"
        elif mode in node._skrub_impl.errors:
            status[node_id] = "error"
        else:
            status[node_id] = "skipped"
    return status


# Some utilities for retrieving the source code for applied functions and
# estimators, and for lines of code in the DataOp's definition stack trace. The
# overall strategy is to get the source lines for a file from linecache, and do
# a small sanity check by comparing the resulting source code with something we
# already have from the DataOp itself: the `__name__` for functions & estimator
# types, or the recorded `line` for DataOp creation stack frame summaries.


def _read_source(source_path):
    lines = linecache.getlines(str(source_path))
    if not lines:
        # e.g. files like <python-input-0> (interactive repl), whose source (in
        # python >= 3.13) is only available from the code object
        raise OSError(f"Could not find source code for {source_path}")
    return lines


def _add_source_file(source_path, source_lines, output_dir):
    source_path = Path(source_path)
    path_hash = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    python_dir = output_dir / "python"
    python_dir.mkdir(exist_ok=True)
    target_file_name = f"{path_hash}.html"
    target_path = python_dir / target_file_name
    if not target_path.is_file():
        page_html = _get_template("python_module.html").render(
            {
                "python_source_code": "".join(source_lines),
                "source_file": str(source_path),
                "module_name": source_path.stem,
            }
        )
        target_path.write_text(page_html, "utf-8")
    return f"python/{target_file_name}"


def _get_source_url(obj, output_dir):
    if isinstance(obj, DataOp):
        return None
    if not callable(obj):
        return None
    try:
        obj = inspect.unwrap(obj)
        # same file as the one used by inspect.findsource (getsourcefile
        # returns None for eg "<...>" files that are not in the linecache)
        source_path = inspect.getsourcefile(obj) or inspect.getfile(obj)
        lines, line_no = inspect.getsourcelines(obj)
        if obj.__name__ == "<lambda>":
            definition = r"\blambda\b"
        else:
            definition = rf"\b(?:def|class)\s+{re.escape(obj.__name__)}\b"
        if not re.search(definition, "".join(lines)):
            # the file or line number do not match the definition of obj, eg
            # it is a func serialized by value in a cloudpickle dump and
            # inspect is returning the source lines for the file where the
            # cloudpickle was loaded, or the file has been modified since obj
            # was defined.
            return None
        source_lines = _read_source(source_path)
        source_file_url = _add_source_file(source_path, source_lines, output_dir)
        return f"{source_file_url}#L{line_no}"
    except Exception:
        return None


def _get_stack_info(stack, output_dir):
    if not stack:
        return []
    result = []
    for frame_summary in traceback.StackSummary.from_list(stack):
        try:
            filename, lineno = frame_summary.filename, frame_summary.lineno
            source_lines = _read_source(filename)
            # e.g. loaded from a pickle: the file may not be the one used to
            # record the stack. Note FrameSummary lines are strip()-ped
            if frame_summary.line != source_lines[lineno - 1].strip():
                raise ValueError("source file does not match the recorded stack")
            source_file_url = _add_source_file(filename, source_lines, output_dir)
            url = f"{source_file_url}#L{lineno}"
        except Exception:
            url = None
        result.append({"url": url, "frame": frame_summary})
    return result


# Objects for which we don't show a docstring or link to source code.
# str, None, PassThrough can come from .skb.apply('passthrough'), .skb.apply(None)
_NO_DOC_OR_SOURCE = (DataOp, BaseChoice, str, type(None), PassThrough)


def _get_doc(obj):
    if isinstance(obj, _NO_DOC_OR_SOURCE):
        return None
    # show the wrapped function's docstring rather than that of the partial class
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.getdoc(obj) or ""


def report(
    data_op,
    environment=None,
    mode="preview",
    clear=True,
    open=True,
    output_dir=None,
    overwrite=False,
    title=None,
    eval=True,
):
    if clear:
        clear_results(data_op, mode)
    try:
        return _make_report(
            data_op,
            environment=environment,
            mode=mode,
            open=open,
            output_dir=output_dir,
            overwrite=overwrite,
            title=title,
            eval=eval,
        )
    finally:
        if clear:
            clear_results(data_op, mode)


def _make_report(
    data_op,
    environment=None,
    mode="preview",
    open=True,
    output_dir=None,
    overwrite=False,
    title=None,
    eval=True,
):
    output_dir = _get_output_dir(output_dir, overwrite)
    if eval:
        try:
            # TODO dump report in callback instead of evaluating full DataOps plan
            # first, so that we can clear intermediate results.
            # See evaluate's `callback` parameter
            result = evaluate(data_op, mode=mode, environment=environment, clear=False)
            evaluate_error = None
        except Exception as e:
            result = None
            evaluate_error = e
    else:
        result = None
        evaluate_error = None
    g = graph(data_op)
    node_status = _node_status(g, mode, eval=eval)
    node_rindex = {id(node): k for k, node in g["nodes"].items()}

    def node_name_to_url(node_name):
        return f"node_{node_name}.html"

    def make_url(node):
        return node_name_to_url(node_rindex[id(node)])

    graph_drawing = draw_data_op_graph(data_op, url=make_url, target="node-frame")
    for node_id, status in node_status.items():
        # graphviz adds the class to the node's <g> in the svg (see data_ops.css)
        if status in ("error", "skipped"):
            graph_drawing.graph.get_node(_dot_id(node_id))[0].set(
                "class", f"{status}-node"
            )
    graph_html = graph_drawing.html_fragment(include_graphviz=True)
    jinja_env = _get_jinja_env()
    index = jinja_env.get_template("index.html").render(
        {
            "graph_html": graph_html,
            "report_title": title,
            "eval": eval,
        }
    )
    index_file = output_dir / "index.html"
    index_file.write_text(index, "utf-8")
    # shown in the index's iframe when no node is selected
    placeholder = jinja_env.get_template("placeholder.html").render(
        {"report_title": title}
    )
    (output_dir / "placeholder.html").write_text(placeholder, "utf-8")

    for i, node in g["nodes"].items():
        report, error, error_msg = None, None, None
        if mode in node._skrub_impl.results:
            report = node_report(node, mode=mode, environment=environment)
        elif mode in node._skrub_impl.errors:
            e = node._skrub_impl.errors[mode]
            error = "".join(_utils.format_exception(e))
            error_msg = "".join(_utils.format_exception_only(e))
            if hasattr(e, "__notes__"):
                error_msg = error_msg.removesuffix("\n".join(e.__notes__) + "\n")
        try:
            metadata = node._skrub_impl.metadata[mode]
        except KeyError:
            # the node was not evaluated
            eval_duration, env_key = None, None
        else:
            eval_duration, env_key = metadata["eval_duration"], metadata["env_key"]
        if isinstance(report, TableReport):
            print(f"Generating report for node {i}")
            report = report.html_snippet()
        node_children = [
            {
                "id": n,
                "description": _utils.simple_repr(g["nodes"][n]),
                "url": node_name_to_url(n),
            }
            for n in g["children"].get(i, [])
        ]
        node_parents = [
            {
                "id": n,
                "description": _utils.simple_repr(g["nodes"][n]),
                "url": node_name_to_url(n),
            }
            for n in g["parents"].get(i, [])
        ]
        source_url = None
        if isinstance(node._skrub_impl, Apply):
            outer_estimator = getattr(
                node._skrub_impl, "estimator_", node._skrub_impl.estimator
            )
            if getattr(node._skrub_impl, "estimator_was_wrapped_", False):
                # unwrap the ApplyToCols
                estimator = outer_estimator.transformer
            else:
                estimator = outer_estimator
            estimator_doc = _get_doc(estimator)
            if isinstance(estimator, _NO_DOC_OR_SOURCE):
                estimator_html_repr = None
                estimator_type = None
            else:
                estimator_type = estimator.__class__.__name__
                try:
                    estimator_html_repr = outer_estimator._repr_html_()
                except Exception:
                    estimator_html_repr = None
                source_url = _get_source_url(estimator.__class__, output_dir)
        else:
            estimator_html_repr = None
            estimator_doc = None
            estimator_type = None
        if isinstance(node._skrub_impl, Call):
            source_url = _get_source_url(node._skrub_impl.func, output_dir)
            applied_func_name = node._skrub_impl.get_func_name()
            applied_func_doc = _get_doc(node._skrub_impl.func)
        else:
            applied_func_name = None
            applied_func_doc = None
        node_page = jinja_env.get_template("node.html").render(
            dict(
                report_title=title,
                total_n_nodes=len(g["nodes"]),
                node_nb=i,
                node_children=node_children,
                node_parents=node_parents,
                node_repr=_utils.simple_repr(node),
                report=report,
                error=error,
                error_msg=error_msg,
                eval_duration=eval_duration,
                env_key=env_key,
                node_creation_stack_info=_get_stack_info(
                    node._skrub_impl.creation_stack(), output_dir
                ),
                node_description=node._skrub_impl.description,
                node_name=node._skrub_impl.name,
                node_uuid=node._skrub_impl.uuid,
                node_type=node._skrub_impl.__class__.__name__,
                is_var=isinstance(node._skrub_impl, Var),
                estimator_type=estimator_type,
                estimator_html_repr=estimator_html_repr,
                estimator_doc=estimator_doc,
                source_url=source_url,
                applied_func_name=applied_func_name,
                applied_func_doc=applied_func_doc,
                eval=eval,
            )
        )
        out = output_dir / f"node_{i}.html"
        out.write_text(node_page, "utf-8")

    index_file = index_file.resolve()
    output = {"result": result, "error": evaluate_error, "report_path": index_file}
    if not open:
        return output
    webbrowser.open(f"file://{index_file}")
    return output


# Font used for node labels when the graph is laid out by the wasm build of
# graphviz in the browser. That build has no fontconfig and cannot measure
# text: it estimates widths from a few well-known family names (Arial,
# Helvetica, Times, Courier) and uses rough defaults for anything else (such as
# the generic "sans-serif" used for native rendering, or a list of families),
# so labels can overflow their boxes. Arial is estimated accurately, and
# browsers render it with Arial or a metric-compatible font (Liberation Sans,
# Arimo). "Helvetica" is estimated just as well but Firefox on Linux does not
# map it to a metric-compatible font. The render_dot_fragment.html template
# appends generic fallbacks to the font-family of the resulting SVG.
_JS_FONT_FAMILY = "Arial"


class GraphDrawing:
    def __init__(self, graph):
        self.graph = graph

    def _dot_for_js(self):
        graph = copy.deepcopy(self.graph)
        for node in graph.get_nodes():
            node.set_fontname(_JS_FONT_FAMILY)
        return graph.to_string()

    def _render_js_template(self, template_name, **kwargs):
        dot = self._dot_for_js().encode("utf-8")
        return _get_template(template_name).render(
            {"dot_base64": base64.b64encode(dot).decode("ascii"), **kwargs}
        )

    @property
    def svg(self):
        _utils.check_graphviz()
        svg = self.graph.create_svg(encoding="utf-8")
        svg = re.sub(b"<title>.*?</title>", b"", svg)
        if "google.colab" in sys.modules:
            # Fix for #1589
            # google colab does not accept <a> without target in svg
            svg = svg.replace(b"<a xlink:title", b'<a target="_blank" xlink:title')
        return svg

    @property
    def png(self):
        _utils.check_graphviz()
        return self.graph.create_png(encoding="utf-8")

    def _repr_html_(self):
        if _utils.has_graphviz():
            return self.svg.decode("utf-8")
        return self._render_js_template("render_dot_iframe.html")

    def html_fragment(self, include_graphviz=False):
        """HTML to insert in a page.

        Without graphviz the graph is drawn in the browser, by a library that is
        either loaded from a CDN or, if `include_graphviz`, included in the
        fragment so that it works without a network connection.
        """
        if _utils.has_graphviz():
            return strip_xml_declaration(self.svg.decode("utf-8"))
        return self._render_js_template(
            "render_dot_fragment.html", include_graphviz=include_graphviz
        )

    @property
    def dot(self):
        return self.graph.to_string()

    @property
    def html(self):
        if _utils.has_graphviz():
            return _get_template("graph.html").render({"svg": self.svg.decode("utf-8")})
        return self._render_js_template("render_dot.html", include_graphviz=True)

    def open(self):
        open_in_browser(self.html)

    def _repr_png_(self):
        return self.png

    def __repr__(self):
        return f"<{self.__class__.__name__}: use .open() to display>"


def _node_kwargs(data_op, *, url=None, target=None, show_ids=False):
    # `url` is a function mapping a node to the address of the page the node
    # links to (graphviz's "URL" attribute: where the link goes). `target` is
    # the name of the browsing context (e.g. an iframe) in which that page is
    # opened (graphviz's "target" attribute: where the link opens). It is only
    # used for nodes that have a link, so it has no effect without `url`.
    impl = data_op._skrub_impl
    label = html.escape(_utils.simple_repr(data_op))
    kwargs = {
        "shape": "box",
        "fontsize": 10,
        "height": 0.25,
        "margin": "0.08,0.06",
        "labelloc": "c",
        "fontname": "sans-serif",
        "color": "black",
    }
    if impl.is_X:
        if not isinstance(impl, SplitX):
            label = f"X: {label}"
            # for SplitX 'X' is already in the repr so prepending it would be redundant
        kwargs["style"] = "filled"
        kwargs["fillcolor"] = "#c6d5f0"
    elif impl.is_y:
        label = f"y: {label}"
        kwargs["style"] = "filled"
        kwargs["fillcolor"] = "#fad9c6"
    if show_ids:
        label = f"{label}\nid: {impl.uuid}"
    if url is not None and (computed_url := url(data_op)) is not None:
        kwargs["URL"] = computed_url
        if target is not None:
            kwargs["target"] = target
        label = label.replace("\n", "<br />")
        label = f'<<FONT COLOR="#1a0dab"><B>{label}</B></FONT>>'
    kwargs["label"] = label
    tooltip = html.escape(impl.creation_stack_last_line())
    if description := impl.description:
        tooltip = f"{tooltip}\n\n{html.escape(description)}"
    # Also escape backslahses inserted in the .dot file
    # otherwise they are interpreted as escape sequences by graphviz
    tooltip = tooltip.replace("\\", "\\\\")
    kwargs["tooltip"] = tooltip
    if isinstance(impl, (Var, Value)):
        kwargs["peripheries"] = 2
    return kwargs


def _dot_id(n):
    return f"node_{n}"


def draw_data_op_graph(
    data_op, *, url=None, target=None, direction="TB", show_ids=False
):
    g = graph(data_op)
    dot_graph = pydot.Dot(rankdir=direction, ranksep=0.4)
    for node_id, e in g["nodes"].items():
        kwargs = _node_kwargs(e, url=url, target=target, show_ids=show_ids)
        kwargs["id"] = _dot_id(node_id)
        node = pydot.Node(_dot_id(node_id), **kwargs)
        dot_graph.add_node(node)
    for c, children in g["children"].items():
        for child in children:
            dot_graph.add_edge(pydot.Edge(_dot_id(child), _dot_id(c), arrowsize=0.7))

    return GraphDrawing(dot_graph)


def describe_params(params, data_op_choices):
    description = {}
    for choice_id, param in params.items():
        choice = data_op_choices["choices"][choice_id]
        choice_name = data_op_choices["choice_display_names"][choice_id]
        if isinstance(choice, Choice):
            # If we have a Choice we use the outcome name if there is one, and
            # if there isn't, the value if it is a simple type otherwise a
            # short repr
            if choice.outcome_names is not None:
                value = choice.outcome_names[param]
            else:
                value = choice.outcomes[param]
                if not isinstance(
                    value, (numbers.Number, bool, str, bytes, type(None))
                ):
                    value = short_repr(value)
        else:
            # If we have a NumericChoice we use the corresponding number. We
            # convert numpy numbers to built-in types to avoid the long
            # 'np.float64(5.0)' repr
            value = param
            if isinstance(value, np.number):
                value = value.tolist()
        description[choice_name] = value
    return description


def describe_param_grid(data_op):
    grid = param_grid(data_op)
    data_op_choices = choice_graph(data_op)

    buf = io.StringIO()
    for subgrid in grid:
        prefix = "- "
        for k, v in subgrid.items():
            assert isinstance(v, (BaseNumericChoice, list))
            choice = data_op_choices["choices"][k]
            name = data_op_choices["choice_display_names"][k]
            buf.write(f"{prefix}{name}: ")
            if isinstance(choice, BaseNumericChoice):
                buf.write(f"{v}\n")
            elif len(v) == 1:
                if choice.outcome_names is not None:
                    buf.write(f"{choice.outcome_names[v[0]]!r}\n")
                else:
                    buf.write(f"{short_repr(choice.outcomes[v[0]])}\n")
            else:
                assert len(v)
                if choice.outcome_names is not None:
                    buf.write(f"{[choice.outcome_names[idx] for idx in v]!r}\n")
                else:
                    buf.write(f"{short_repr([choice.outcomes[idx] for idx in v])}\n")
            prefix = "  "
    return buf.getvalue() or "<empty parameter grid>\n"
