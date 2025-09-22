import datetime
import html
import io
import numbers
import re
import shutil
import sys
import webbrowser
from pathlib import Path

import jinja2
import numpy as np
from sklearn.base import BaseEstimator

from .. import _dataframe as sbd
from .. import datasets
from .._config import get_config
from .._reporting import TableReport
from .._reporting._serve import open_in_browser
from .._utils import Repr, format_duration, random_string, short_repr
from . import _utils
from ._choosing import BaseNumericChoice, Choice
from ._data_ops import Apply, Value, Var
from ._evaluation import choice_graph, clear_results, evaluate, graph, param_grid
from ._subsampling import uses_subsampling


def _get_jinja_env():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            Path(__file__).resolve().parents[1]
            / "_reporting"
            / "_data"
            / "templates"
            / "data_ops",
            encoding="UTF-8",
        ),
        autoescape=True,
    )
    env.filters["format_duration"] = format_duration
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
        output_dir = (
            datasets.get_data_dir()
            / "execution_reports"
            / f"full_data_op_report_{now}_{random_string()}"
        )
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


def _node_status(data_op_graph, mode):
    status = {}
    for node_id, node in data_op_graph["nodes"].items():
        if mode in node._skrub_impl.results:
            status[node_id] = "success"
        elif mode in node._skrub_impl.errors:
            status[node_id] = "error"
        else:
            status[node_id] = "none"
    return status


def full_report(
    data_op,
    environment=None,
    mode="preview",
    clear=True,
    open=True,
    output_dir=None,
    overwrite=False,
):
    if clear:
        clear_results(data_op, mode)
    try:
        return _make_full_report(
            data_op,
            environment=environment,
            mode=mode,
            open=open,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    finally:
        if clear:
            clear_results(data_op, mode)


def _make_full_report(
    data_op,
    environment=None,
    mode="preview",
    open=True,
    output_dir=None,
    overwrite=False,
):
    _check_graphviz()
    output_dir = _get_output_dir(output_dir, overwrite)
    try:
        # TODO dump report in callback instead of evaluating full DataOps plan
        # first, so that we can clear intermediate results.
        # See evaluate's `callback` parameter
        result = evaluate(data_op, mode=mode, environment=environment, clear=False)
        evaluate_error = None
    except Exception as e:
        result = None
        evaluate_error = e
    g = graph(data_op)
    node_status = _node_status(g, mode)
    node_rindex = {id(node): k for k, node in g["nodes"].items()}

    def node_name_to_url(node_name):
        return f"node_{node_name}.html"

    def make_url(node):
        return node_name_to_url(node_rindex[id(node)])

    svg = draw_data_op_graph(data_op, url=make_url).svg.decode("utf-8")
    jinja_env = _get_jinja_env()
    index = jinja_env.get_template("index.html").render(
        {"svg": svg, "node_status": node_status}
    )
    index_file = output_dir / "index.html"
    index_file.write_text(index, "utf-8")

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
            eval_duration = node._skrub_impl.metadata[mode]["eval_duration"]
        except KeyError:
            # the node was not evaluated
            eval_duration = None
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
        if isinstance(node._skrub_impl, Apply):
            estimator = getattr(
                node._skrub_impl, "estimator_", node._skrub_impl.estimator
            )
            if isinstance(estimator, BaseEstimator):
                estimator_html_repr = estimator._repr_html_()
            else:
                estimator_html_repr = None
        else:
            estimator_html_repr = None
        node_page = jinja_env.get_template("node.html").render(
            dict(
                total_n_nodes=len(g["nodes"]),
                node_nb=i,
                node_children=node_children,
                node_parents=node_parents,
                node_repr=_utils.simple_repr(node),
                report=report,
                error=error,
                error_msg=error_msg,
                eval_duration=eval_duration,
                node_creation_stack_description=node._skrub_impl.creation_stack_description(),
                node_description=node._skrub_impl.description,
                node_name=node._skrub_impl.name,
                node_type=node._skrub_impl.__class__.__name__,
                svg=svg,
                node_status=node_status,
                estimator_html_repr=estimator_html_repr,
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


class GraphDrawing:
    def __init__(self, graph):
        self.graph = graph

    @property
    def svg(self):
        svg = self.graph.create_svg(encoding="utf-8")
        svg = re.sub(b"<title>.*?</title>", b"", svg)
        if "google.colab" in sys.modules:
            # Fix for #1589
            # google colab does not accept <a> without target in svg
            svg = svg.replace(b"<a xlink:title", b'<a target="_blank" xlink:title')
        return svg

    @property
    def png(self):
        return self.graph.create_png(encoding="utf-8")

    def _repr_html_(self):
        return self.svg.decode("utf-8")

    def open(self):
        open_in_browser(
            _get_template("graph.html").render({"svg": self.svg.decode("utf-8")})
        )

    def _repr_png_(self):
        return self.png

    def __repr__(self):
        return f"<{self.__class__.__name__}: use .open() to display>"


def _node_kwargs(data_op, url=None):
    label = html.escape(_utils.simple_repr(data_op))
    kwargs = {
        "shape": "box",
        "fontsize": 12,
        "height": 0.3,
        "margin": "0.1,0.08",
        "labelloc": "c",
        "fontname": "sans-serif",
        "color": "black",
    }
    if data_op._skrub_impl.is_X:
        label = f"X: {label}"
        kwargs["style"] = "filled"
        kwargs["fillcolor"] = "#c6d5f0"
    elif data_op._skrub_impl.is_y:
        label = f"y: {label}"
        kwargs["style"] = "filled"
        kwargs["fillcolor"] = "#fad9c6"
    if url is not None and (computed_url := url(data_op)) is not None:
        kwargs["URL"] = computed_url
        label = label.replace("\n", "<br />")
        label = f'<<FONT COLOR="#1a0dab"><B>{label}</B></FONT>>'
    kwargs["label"] = label
    tooltip = html.escape(data_op._skrub_impl.creation_stack_last_line())
    if description := data_op._skrub_impl.description:
        tooltip = f"{tooltip}\n\n{html.escape(description)}"
    kwargs["tooltip"] = tooltip
    if isinstance(data_op._skrub_impl, (Var, Value)):
        kwargs["peripheries"] = 2
    return kwargs


def _dot_id(n):
    return f"node_{n}"


def _has_graphviz():
    try:
        import pydot

        g = pydot.Dot()
        g.add_node(pydot.Node("node 0"))
        g.create_svg()
        return True
    except Exception:
        return False


def _check_graphviz():
    if _has_graphviz():
        return
    raise ImportError("Please install pydot and graphviz to draw data_op graphs.")


def draw_data_op_graph(data_op, url=None, direction="TB"):
    # TODO if pydot or graphviz not available fallback on some other plotting
    # solution eg a vendored copy of mermaid? outputting html instead of svg
    _check_graphviz()

    import pydot

    g = graph(data_op)
    dot_graph = pydot.Dot(rankdir=direction)
    for node_id, e in g["nodes"].items():
        kwargs = _node_kwargs(e, url=url)
        kwargs["id"] = _dot_id(node_id)
        node = pydot.Node(_dot_id(node_id), **kwargs)
        dot_graph.add_node(node)
    for c, children in g["children"].items():
        for child in children:
            dot_graph.add_edge(pydot.Edge(_dot_id(child), _dot_id(c)))

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
