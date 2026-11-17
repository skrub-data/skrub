from skrub import tabular_pipeline


def generate_demo():
    """Write the estimator diagram shown on the homepage.

    ``_repr_html_`` returns a full HTML document (``<style>`` + ``<body>`` +
    ``<script>``); we drop the ``<body>`` tags so the snippet can be included
    inside the landing page.
    """
    html = tabular_pipeline("regressor")._repr_html_()
    html = html.replace("<body>", "").replace("</body>", "")

    with open(
        "_templates/demo_tabular_pipeline_generated.html", "w", encoding="utf-8"
    ) as f:
        f.write(html)
