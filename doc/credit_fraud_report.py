import importlib.util
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path


def create_credit_fraud_report():
    """Generate the credit fraud report and the graph shown on the homepage."""
    output_dir = (
        Path(__file__).parent / "_build" / "html" / "_static" / "credit_fraud_report"
    )
    if (output_dir / "index.html").exists() and (
        output_dir / "credit_fraud_graph.svg"
    ).exists():
        return output_dir

    doc_dir = Path(__file__).parent
    # Load the example with a relative path, from the doc directory, so that the
    # source locations embedded in the report and the graph (shown as tooltips)
    # stay relative instead of leaking absolute paths into the published page.
    module_path = (
        Path("credit_fraud_example_for_home_page") / "credit_fraud_with_choices.py"
    )

    cwd = Path.cwd()
    os.chdir(doc_dir)
    try:
        loader = SourceFileLoader("credit_fraud_with_choices", str(module_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        module.create_report(output_dir)
    finally:
        os.chdir(cwd)

    return output_dir
