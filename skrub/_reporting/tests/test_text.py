from skrubview._summarize import summarize_dataframe
from skrubview._text import to_text


def test_to_text(make_dataframe):
    df = make_dataframe()
    summary = summarize_dataframe(df)
    text = to_text(summary)
    assert f"{df.shape[0]} rows and {df.shape[1]} columns" in text
