"""Command-line interface to generate reports."""

import argparse
from pathlib import Path

from ._table_report import TableReport
from ._utils import read


def _get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate a skrub report for a CSV or Parquet file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help=(
            "CSV or Parquet file for which a report will be generated. "
            "The filename extension must be '.csv' or '.parquet' . "
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Output file in which to store the report. If not provided, "
            "the report is opened in a web browser and no file is created."
        ),
    )
    parser.add_argument(
        "--order_by",
        type=str,
        default=None,
        help=(
            "Sort by this column. Other numerical columns will be plotted as functions"
            " of the sorting column. Must be of numerical or datetime type."
        ),
    )
    return parser


def make_report(raw_args=None):
    parser = _get_arg_parser()
    args = parser.parse_args(raw_args)
    input_file = Path(args.input_file)
    df = read(input_file)
    report = TableReport(df, order_by=args.order_by, title=input_file.name)
    if args.output is not None:
        out = Path(args.output)
        out.write_text(report.html(), encoding="utf-8")
        print(f"The report has been saved in '{out}'.")
    else:
        report.open()
