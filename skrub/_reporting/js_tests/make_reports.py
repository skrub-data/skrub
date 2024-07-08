import pathlib

from skrub import TableReport, datasets

reports_dir = pathlib.Path(__file__).resolve().parent / "_reports"
reports_dir.mkdir(exist_ok=True)

df = datasets.fetch_employee_salaries().X
html = TableReport(df).html()
(reports_dir / "employee_salaries.html").write_text(html, encoding="UTF-8")
