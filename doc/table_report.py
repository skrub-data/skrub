from skrub import TableReport
from skrub.datasets import fetch_employee_salaries


def generate_demo():
    X = fetch_employee_salaries().X
    with open("_templates/demo_table_report_generated.html", "w") as f:
        f.write(TableReport(X).html_snippet())
