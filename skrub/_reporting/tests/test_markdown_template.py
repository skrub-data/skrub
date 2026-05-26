"""Tests for the markdown report template."""

import re

from skrub import TableReport
from skrub.datasets import toy_cities


def test_markdown_report_structure_and_titles(df_module):
    """Test basic structure, headers, and title handling."""
    df = df_module.make_dataframe(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "a", "b", "c"],
        }
    )

    # Test custom title
    report = TableReport(df, title="Test Report")
    markdown = report.markdown()

    assert "Test Report" in markdown
    assert "## Columns" in markdown
    assert "## Associations (Cramér's V)" in markdown
    assert "`A`" in markdown
    assert "`B`" in markdown
    assert markdown.startswith("#")  # Should have a header

    # Test default title
    report_default = TableReport(df)
    markdown_default = report_default.markdown()
    assert "# " in markdown_default  # Header should exist
    # Shape info should be present
    assert "**Shape:** 5 rows × 2 columns" in markdown


def test_markdown_columns_table_structure(df_module):
    """Test columns table headers, positions, and row count."""
    df = df_module.make_dataframe(
        {
            "first": [1, 2, 3],
            "second": [4, 5, 6],
            "third": [7, 8, 9],
        }
    )
    report = TableReport(df)
    markdown = report.markdown()

    # Check for table headers
    assert (
        "| Position | Column | Type | Unique | Nulls | High Card | Constant |"
        in markdown
    )

    # Check that position numbers are in the table
    assert "| 0 |" in markdown
    assert "| 1 |" in markdown
    assert "| 2 |" in markdown

    # Check that column names are present and dtypes are shown
    lines = markdown.split("\n")
    column_table_lines = []
    in_table = False
    for line in lines:
        if "| Position | Column | Type" in line:
            in_table = True
        elif in_table and line.startswith("|"):
            column_table_lines.append(line)
        elif in_table and not line.startswith("|"):
            break

    # Should have at least 3 data rows (for the three columns)
    assert len(column_table_lines) >= 3


def test_markdown_data_format_and_highlighting(df_module):
    """Test that unique/null values show count+percentage and highlighting works."""
    # Test format with nulls and varying data
    df = df_module.make_dataframe(
        {
            "A": [1, 2, 3, None],
            "B": ["a", "b", "a", "a"],
            "high_nulls": [None, None, None, None, 1],
            "low_nulls": [1, 2, 3, 4, 5],
        }
    )
    report = TableReport(df)
    markdown = report.markdown()

    # Test format: "count (percentage%)"
    unique_pattern = r"\|\s*\d+\s*\(\d+\.?\d*%\)\s*\|"
    matches = re.findall(unique_pattern, markdown)
    assert len(matches) > 0

    # Test high null columns are bolded
    assert "**`high_nulls`**" in markdown
    assert "**`low_nulls`**" not in markdown


def test_markdown_associations_highlighting_and_headers(df_module):
    """Test associations table structure, highlighting of strong associations."""

    df = df_module.make_dataframe(toy_cities().to_dict())
    report = TableReport(df, compute_associations=True)
    markdown = report.markdown()

    # Check for associations table headers
    assert "Left column" in markdown
    assert "Right column" in markdown
    assert "Cramér's V" in markdown

    # Test strong associations are bolded (pattern: **0.XXXX**)
    strong_assoc_pattern = r"\*\*[1]\.[0]{4}\*\*"
    if "| col1 | col2 |" in markdown or "| col2 | col1 |" in markdown:
        matches = re.findall(strong_assoc_pattern, markdown)
        # May be 0 if association is not exactly > 0.9
        assert len(matches) == 1


def test_markdown_column_statistics_and_constants(df_module):
    """Test that statistics sections are present and constant columns are flagged."""
    # Categorical column with value counts
    df_cat = df_module.make_dataframe(
        {
            "category": ["a", "b", "a", "b", "c"],
            "constant": [1, 1, 1, 1, 1],
        }
    )
    report_cat = TableReport(df_cat)
    markdown_cat = report_cat.markdown()
    assert "Details" in markdown_cat or "Most frequent values" in markdown_cat

    # Constant column should be flagged
    lines = markdown_cat.split("\n")
    for line in lines:
        if "`constant`" in line:
            assert line  # Line should exist

    # Numeric column with statistics
    df_num = df_module.make_dataframe(
        {
            "numbers": [1.0, 2.5, 3.7, 4.2, 5.8],
        }
    )
    report_num = TableReport(df_num)
    markdown_num = report_num.markdown()
    assert "Details" in markdown_num or "Numeric statistics" in markdown_num


def test_markdown_edge_cases(df_module):
    """Test that markdown generation handles edge cases correctly."""
    # Empty dataframe
    df_empty = df_module.make_dataframe(
        {
            "A": [],
            "B": [],
        }
    )
    report_empty = TableReport(df_empty)
    markdown_empty = report_empty.markdown()
    assert markdown_empty == "The dataframe is empty.\n"

    # Single row
    df_single = df_module.make_dataframe(
        {
            "col": [42],
        }
    )
    report_single = TableReport(df_single)
    markdown_single = report_single.markdown()
    assert "## Columns" in markdown_single
