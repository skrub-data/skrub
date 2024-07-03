from skrub._reporting._html import to_html
from skrub._reporting._summarize import summarize_dataframe


def test_to_html(df_module):
    df = df_module.example_dataframe
    summary = summarize_dataframe(df)
    to_html(summary)
