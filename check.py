import webbrowser
from pathlib import Path

from skrub import TableVectorizer, datasets

data = datasets.fetch_employee_salaries()
X, y = data.X, data.y

tv = TableVectorizer(high_cardinality="passthrough")

tv.fit(X, y)

print("hi")

html = tv._repr_html_()
page = f"""
<html>
<head>
</head>
<body>
{html}
</body>
</html>
"""

Path(".", "repr.html").write_text(page, "utf-8")
webbrowser.open_new_tab("repr.html")
