from skrub import TableReport
from skrub.datasets import fetch_employee_salaries


def generate_demo():
    X = fetch_employee_salaries().X
    X = X.sample(frac=1, random_state=145).reset_index(drop=True)

    with open(
        "_templates/demo_table_report_generated.html", "w", encoding="utf-8"
    ) as f:
        f.write(TableReport(X, n_rows=5).html_snippet())
