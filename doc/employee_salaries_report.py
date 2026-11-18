from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor

import skrub
import skrub.datasets
from skrub import TableVectorizer


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
