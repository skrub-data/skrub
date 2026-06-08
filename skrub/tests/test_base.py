import re

from sklearn.utils import estimator_html_repr

from skrub import (
    ApplyToCols,
    Cleaner,
    DropCols,
    SelectCols,
    StringEncoder,
    TableVectorizer,
)


def test_doc_link_apply_to_cols():
    """The wrapped transformer's doc link appears in the HTML repr of ApplyToCols."""
    html = estimator_html_repr(ApplyToCols(StringEncoder()))
    links = set(re.findall(r'href="(https?://[^#"]+)"', html))
    assert (
        "https://skrub-data.org/stable/reference/generated/skrub.StringEncoder.html"
        in links
    )

    html = estimator_html_repr(ApplyToCols(TableVectorizer()))
    links = set(re.findall(r'href="(https?://[^#"]+)"', html))
    assert (
        "https://skrub-data.org/stable/reference/generated/skrub.TableVectorizer.html"
        in links
    )


def test_doc_link_skrub_class_select_cols():
    """Public skrub classes get a link to skrub documentation."""
    link = SelectCols(cols=[])._get_doc_link()
    assert link == (
        "https://skrub-data.org/stable/reference/generated/skrub.SelectCols.html"
    )
    link = DropCols(cols=[])._get_doc_link()
    assert link == (
        "https://skrub-data.org/stable/reference/generated/skrub.DropCols.html"
    )


def test_doc_link_table_vectorizer():
    """Public skrub classes get a link to skrub documentation."""
    link = TableVectorizer()._get_doc_link()
    assert link == (
        "https://skrub-data.org/stable/reference/generated/skrub.TableVectorizer.html"
    )
    link = Cleaner()._get_doc_link()
    assert link == (
        "https://skrub-data.org/stable/reference/generated/skrub.Cleaner.html"
    )
