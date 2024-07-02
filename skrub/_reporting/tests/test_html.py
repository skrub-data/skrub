from skrub._reporting._summarize import summarize_dataframe
from skrub._reporting._html import to_html


def test_to_html(df_module):
    df = df_module.example_dataframe
    summary = summarize_dataframe(df)
    html = to_html(summary)
