import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor

import skrub
import skrub.datasets
from skrub import TableVectorizer


def create_credit_fraud_report():
    """Generate the credit fraud report and the graph shown on the homepage.

    Both come from the same example script so that the graph, the interactive
    report and the code snippet on the homepage always describe the same
    pipeline.
    """
    output_dir = (
        Path(__file__).parent / "_build" / "html" / "_static" / "credit_fraud_report"
    )
    if (output_dir / "index.html").exists():
        return output_dir

    doc_dir = Path(__file__).parent
    # Run from the doc directory with a relative path so that the source
    # locations embedded in the graph (shown as tooltips) stay relative
    # instead of leaking absolute paths into the published page.
    script = Path("credit_fraud_example_for_home_page") / "credit_fraud_with_choices.py"

    with tempfile.TemporaryDirectory() as tmp_dir:
        argv, cwd = sys.argv, Path.cwd()
        sys.argv = [str(script), "--output_dir", tmp_dir]
        os.chdir(doc_dir)
        try:
            runpy.run_path(str(script), run_name="__main__")
        finally:
            sys.argv = argv
            os.chdir(cwd)

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(tmp_dir) / "report", output_dir, dirs_exist_ok=True)
        shutil.copyfile(
            Path(tmp_dir) / "credit_fraud_graph.svg",
            output_dir / "credit_fraud_graph.svg",
        )

    return output_dir


def create_employee_salaries_report():
    output_dir = (
        Path(__file__).parent
        / "_build"
        / "html"
        / "_static"
        / "employee_salaries_report"
    )
    if (output_dir / "index.html").exists():
        return output_dir

    pred = (
        skrub.var("employee_data")
        .skb.apply(TableVectorizer())
        .skb.apply(HistGradientBoostingRegressor(), y=skrub.var("salary"))
    )

    dataset = skrub.datasets.fetch_employee_salaries(split="train")

    pred.skb.report(
        {"employee_data": dataset.X, "salary": dataset.y},
        output_dir=output_dir,
        overwrite=True,
        open=False,
    )

    return output_dir


def create_data_ops_report():
    credit_fraud_folder = create_credit_fraud_report()
    employee_salary_folder = create_employee_salaries_report()

    assert (credit_fraud_folder / "index.html").exists()
    assert (credit_fraud_folder / "credit_fraud_graph.svg").exists()
    assert (employee_salary_folder / "index.html").exists()
