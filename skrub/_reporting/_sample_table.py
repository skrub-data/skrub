"""
Prepare data for the HTML table in the "sample" tab panel of the TableReport.

This module builds a dictionary containing the information used to populate the
sample table: the actual sample data and some metadata used for interactions
(such as navigation with the arrow keys, copy-pasting, showing the selected
column's histogram).

The positions of table cells in the HTML table are indexed by row i, column j.
Those correspond to positions in the table itself, not in the dataframe.
i grows downwards and j towards the right.

The first dataframe data cell (df.iloc[0, 0]) is at i=0, j=0. Thus the table
head corresponds to negative i; the pandas index (if any) has negative j.

In the simple case of a Polars dataframe which has no index and just one row of
column names, the indices look like:

           j=0           j=1          j=2
i=-1 | Country name | Population |    Capital    |
--------------------------------------------------
i=0  | Paraguay     | 6M         | Asunción      |
i=1  | Ivory Coast  | 31M        | Yamoussoukro  |

Pandas is slightly more complicated due to the possible presence of
MultiIndexes, see the docstring of `_PandasTable` for details.

The name `column_idx` refers to the index (position) of a column in the
dataframe. In practice it coincides with j for j >= 0.

`make_table` returns the table, which is a dictionary with the following keys:

Table
  start_i: smallest i in the table, eg -1 in the example above
  stop_i: one past the largest i in the table, eg 2 in the example above
  start_j: smallest j in the table, eg 0 in the example above
  stop_j: one past the largest j in the table, eg 3 in the example above
  parts: a list of table parts

"parts" is a python list where each item is a dictionary with the following keys:

Table Part
  name:
    a name for the part, one of "thead", "top_slice" (the tbody containing the
    first few rows), "ellipsis" (optional, the row of " ⋮ " indicating that the
    table is truncated), "bottom_slice" (optional, tbody containing the last
    few rows).
  elem: the name of a html element: "thead" (table head) or "tbody" (table body)
  rows: list of table rows

"rows" is a python list where each item is itself a list, representing a table
row and containing table cells. Each table cell (ie each item in
`table["parts"][0]["rows"][0]`) is a dictionary with the following keys:

Table Cell
  i, j: the position of the cell in the table
  column_idx: the corresponding column in the dataframe, if any
  value: the content of the cell (the corresponding value in the dataframe)
  elem: the name of a html element, "th" (table header) or "td" (table data cell)
  role:
    a semantic description of this cell's content, such as "index-level-name",
    "dataframe-data" or "padding"
  rowspan, colspan, scope: the values for the html attributes with the same names
"""

import pandas as pd

from .. import _dataframe as sbd
from .._dispatch import dispatch, raise_dispatch_unregistered_type

__all__ = ["make_table"]


@dispatch
def make_table(df, max_top_slice_size=5, max_bottom_slice_size=5):
    """Create the data for the sample table in the first report tab panel.

    Parameters
    ----------
    df: dataframe
    max_top_slice_size: int
      The maximum number of rows from the top of the dataframe to display in
      the first part of the table (before the row of "...").
    max_bottom_slice_size: int
      Maximum number of rows from the bottom of the dataframe to display in the
      last part of the table (below "...").

    Returns
    -------
    A dictionary with all the information needed by the html template, see this
    module's docstring for details.
    """
    raise_dispatch_unregistered_type(df, kind="DataFrame")


@make_table.specialize("pandas", argument_type="DataFrame")
def _make_table_pandas(df, max_top_slice_size=5, max_bottom_slice_size=5):
    return _PandasTable(df, max_top_slice_size, max_bottom_slice_size).table_data


@make_table.specialize("polars", argument_type="DataFrame")
def _make_table_polars(df, max_top_slice_size=5, max_bottom_slice_size=5):
    return _PolarsTable(df, max_top_slice_size, max_bottom_slice_size).table_data


def _pick_slice_sizes(df, max_top_size, max_bottom_size):
    """Return (top slice size, bottom slice size, is elided).

    If the whole dataframe fits in max_top_size + max_bottom_size, the whole
    dataframe is shown as one table part (there are no ellipsis nor bottom
    slice).

    is_elided indicates whether some rows of the dataframe will be missing
    from the table (and thus a row o f"..." needs to be shown).
    """
    if sbd.shape(df)[0] <= max_top_size + max_bottom_size:
        return sbd.shape(df)[0], 0, False
    return max_top_size, max_bottom_size, True


class _PolarsTable:
    """Helper for _make_table_polars."""

    def __init__(self, df, max_top_slice_size, max_bottom_slice_size):
        self.df = df
        self.max_top_slice_size = max_top_slice_size
        self.max_bottom_slice_size = max_bottom_slice_size

        self.start_i = -1
        self.start_j = 0

        self.top_slice_size, self.bottom_slice_size, self.is_elided = _pick_slice_sizes(
            self.df, self.max_top_slice_size, self.max_bottom_slice_size
        )

        self.parts = []
        self.add_table_head()
        self.add_df_slices()
        self.table_data = {
            "parts": self.parts,
            "start_i": self.start_i,
            "start_j": self.start_j,
            "stop_j": int(sbd.shape(df)[1]),
            "stop_i": int(self.top_slice_size + self.bottom_slice_size),
        }

    def add_table_head(self):
        table_row = [
            {
                "value": c,
                "i": -1,
                "j": j,
                "elem": "th",
                "scope": "column",
                "role": "column-name",
                "column_idx": j,
            }
            for j, c in enumerate(sbd.column_names(self.df))
        ]
        self.parts.append({"name": "thead", "elem": "thead", "rows": [table_row]})

    def add_df_slices(self):
        self.add_table_body(sbd.slice(self.df, self.top_slice_size), "top_slice", 0)
        if self.is_elided:
            self.add_ellipsis()
        if self.bottom_slice_size:
            self.add_table_body(
                sbd.slice(self.df, -self.bottom_slice_size, sbd.shape(self.df)[0]),
                "bottom_slice",
                self.top_slice_size,
            )

    def add_table_body(self, sub_df, part_name, start_i):
        tbody = {"name": part_name, "elem": "tbody", "rows": []}
        self.parts.append(tbody)
        for tr_count, df_row in enumerate(sub_df.iter_rows()):
            i = start_i + tr_count
            tr = [
                {
                    "value": v,
                    "i": i,
                    "j": j,
                    "elem": "td",
                    "column_idx": j,
                    "role": "dataframe-data",
                }
                for j, v in enumerate(df_row)
            ]
            tbody["rows"].append(tr)

    def add_ellipsis(self):
        self.parts.append({"name": "ellipsis", "elem": "tbody"})


class _LevelCounter:
    """Helper to build the spans of table headers for a pandas multi-index.

    The table headers (th) that represent values of a level of a pandas
    multi-index can span multiple table columns (or rows, depending on the
    orientation -- whether we are dealing with `df.columns` or `df.index`).

    Each `_LevelCounter` manages one level and starts a new header when the
    value of that level changes, and increments the span otherwise.

    The user of `_LevelCounter` (`_multi_index_table_headers`) is responsible
    to ensure that:

      - Whenever a level starts a new header, all levels that are lower in the
        multi-index hierarchy also start a new header.
      - The last (innermost) level starts a new header for each multi-index
        item (ie for each dataframe row or column).

    To do so, `start_new_th` allows forcing closing the current table header if
    it exists and opening a new one.
    """

    def __init__(self, level_nb):
        self.level_nb = level_nb
        self.current_th = None

    def start_new_th(self):
        self.current_th = None

    def get_th(self, index_item):
        """Get a table header.

        If no new header is needed, the "span" of the previously returned one
        is incremented and `get_th` returns `None`.
        """
        value = index_item[self.level_nb]
        if self.current_th is not None and self.current_th["value"] == value:
            self.current_th["span"] += 1
            return None
        self.current_th = {"value": value, "span": 1}
        return self.current_th


def _to_multi(pd_index):
    """Convert any pandas index to MultiIndex."""
    if isinstance(pd_index, pd.MultiIndex):
        return pd_index
    return pd.MultiIndex.from_arrays(
        [pd_index], names=[n] if (n := pd_index.name) is not None else None
    )


def _multi_index_table_headers(idx, orientation):
    """Prepare the table headers (th) that represent items in a pandas multi-index.

    `idx` is the pandas multi-index (`df.index` or `df.columns`).

    `orientation` is the orientation in which the result will be displayed:
    "horizontal" means each multi-index level will be a row in the HTML table
    (used for `df.columns`), "vertical" means each level will be a colun in the
    HTML table (used for `df.index`).
    """
    idx = _to_multi(idx)
    # See `_LevelCounter` docstring for details
    counters = [_LevelCounter(i) for i in range(len(idx.levels))]

    # we first build the table rows assuming "vertical" orientation then
    # transpose them at the end if the orientation is "horizontal".
    table_rows = []
    for idx_item in idx.to_flat_index():
        tr = []
        table_rows.append(tr)
        # The innermost level has one th for each dataframe column or row, ie
        # it never spans more than 1 table column or row: we start a new th for
        # each index item.
        counters[-1].start_new_th()
        for i in range(len(counters)):
            c = counters[i]
            # get_th either returns a new th or increments the span of the
            # previously returned one and returns None
            cell = c.get_th(idx_item)
            if cell is not None:
                # If this level starts a new th, all levels that are lower in
                # the hierarchy must be reset as well.
                for child_c in counters[i + 1 :]:
                    child_c.start_new_th()
            tr.append(cell)

    if orientation == "horizontal":
        table_rows = list(zip(*table_rows))
        span_name = "colspan"
    else:
        assert orientation == "vertical"
        span_name = "rowspan"
    # rename the "span" to either "rowspan" or "colspan" depending on the
    # orientation.
    for r in table_rows:
        for cell in r:
            if cell is not None:
                cell[span_name] = cell.pop("span")
    return table_rows


def _n_levels(pd_index):
    if (levels := getattr(pd_index, "levels", None)) is not None:
        return len(levels)
    return 1


def _level_names(pd_index):
    if (names := getattr(pd_index, "names", None)) is not None:
        return names
    if (name := getattr(pd_index, "name", None)) is not None:
        return [name]
    return [None] * _n_levels(pd_index)


class _PandasTable:
    """Helper for _make_table_pandas.

    Anywhere in the code, i, j refer to positions in the HTML table --
    not in the dataframe. i grows downwards and j grows rightwards.
    The first data cell in the dataframe df.iloc[0, 0] goes at i=0, j=0
    The table head (column level names) thus corresponds to negative i,
    and the index (index level names) to negative j.
    for table with 2 column levels named week and day, and 2 index levels
    named building and floor, the table head looks like:

             j=-2        j=-1      j=0             j=1          j=2   j=3
    i=-3 |           week     |              w1               |    w2     |
    i=-2 |           day      |      mon      |       tue     | mon | tue |
    i=-1 | building   | floor |               |               |     |     |
    -------------------------- end of thead -----------------------------------
    i=0  | my house   | 1st   | df.iloc[0, 0] | df.iloc[0, 1] | ... | ... |
    i=1  |            | 2nd   | df.iloc[1, 0] | df.iloc[1, 1] | ... | ... |
    i=2  | your house | 1st   | ...           | ...           | ... | ... |
    i=3  |            | 2nd   | ...           | ...           | ... | ... |


    start_i, start_j are the first i, j coords (here -3, -2)
    stop_i, stop_j are one past the last i, j coords (df.shape[0], df.shape[1])
    """

    def __init__(self, df, max_top_slice_size, max_bottom_slice_size):
        self.df = df
        self.max_top_slice_size = max_top_slice_size
        self.max_bottom_slice_size = max_bottom_slice_size

        self.n_col_levels = _n_levels(self.df.columns)
        self.n_idx_levels = _n_levels(self.df.index)
        self.start_i = -_n_levels(self.df.columns) - 1
        self.start_j = -_n_levels(self.df.index)

        self.top_slice_size, self.bottom_slice_size, self.is_elided = _pick_slice_sizes(
            self.df, self.max_top_slice_size, self.max_bottom_slice_size
        )

        self.parts = []
        self.add_table_head()
        self.add_df_slices()
        self.table_data = {
            "parts": self.parts,
            "start_i": self.start_i,
            "start_j": self.start_j,
            "stop_j": int(df.shape[1]),
            "stop_i": int(self.top_slice_size + self.bottom_slice_size),
        }

    def add_table_head(self):
        table_headers = _multi_index_table_headers(
            self.df.columns, orientation="horizontal"
        )
        table_head = {"name": "thead", "elem": "thead", "rows": []}
        self.parts.append(table_head)
        # We first build the thead without the top-left corner (columns and
        # index level names), then add the corner at the end. During the first
        # phase the trs are shorter than they will be (they are missing the
        # first few table headers)
        for i, table_row_data in zip(range(self.start_i, -1), table_headers):
            table_row = []
            for j, th in enumerate(table_row_data):
                if th is not None:
                    th["i"] = i
                    th["j"] = j
                    th["elem"] = "th"
                    th["scope"] = "column"
                    th["role"] = "columns-level-value"
                    if i == -2:
                        # The last row has a row span of 2 to make room for the
                        # index level names which we will later add at position
                        # i = -1 .
                        th["rowspan"] = 2
                        # This corresponds to the innermost df columns level,
                        # so each header here corresponds to an actual
                        # dataframe column. We store the dataframe column
                        # position (idx) which is used to display the
                        # corresponding card (which contains the plots &
                        # summary stats for this df column).
                        th["column_idx"] = j

                    table_row.append(th)
            table_head["rows"].append(table_row)
        # The row that will contain index level names if there are any
        table_head["rows"].append([])
        self.add_table_head_corner()

    def add_table_head_corner(self):
        # Whether we have some column and index level names to display. When
        # they are all None we don't show them or create table cells for them.
        has_idx_levels = any(n is not None for n in _level_names(self.df.index))
        has_col_levels = any(n is not None for n in _level_names(self.df.columns))
        thead_rows = self.parts[0]["rows"]
        if not has_idx_levels and not has_col_levels:
            # When we have nothing to display we just insert a big empty cell
            # to fill the space at the beginning of the first row of the first
            # part (the table head).
            thead_rows[0].insert(
                0,
                {
                    "i": self.start_i,
                    "j": self.start_j,
                    "rowspan": -self.start_i,
                    "colspan": -self.start_j,
                    "elem": "th",
                    "value": None,
                    "role": "padding",
                },
            )
            return
        if has_col_levels:
            # insert headers that span horizontally over all index levels,
            # containing the name of the column levels.
            for i, level_name, tr in zip(
                range(self.start_i, -1), _level_names(self.df.columns), thead_rows
            ):
                tr.insert(
                    0,
                    {
                        "colspan": -self.start_j,
                        "rowspan": 2 if i == -2 else 1,
                        "i": i,
                        "j": self.start_j,
                        "value": level_name,
                        "elem": "th",
                        "scope": "row",
                        "role": "columns-level-name",
                    },
                )
        if has_idx_levels:
            self.insert_index_level_names(thead_rows, has_col_levels)

    def insert_index_level_names(self, thead_rows, has_col_levels):
        if has_col_levels:
            # if there are column level names, insert the index level names
            # below the column level names (at i=-1), spanning 1 row
            i, rowspan = -1, 1
        else:
            # if there are no column level names, the index level names are at
            # the top-left cell and the row span is from -start_i to the first
            # data row (i=0).
            i, rowspan = self.start_i, -self.start_i
        tr = []
        for j, level_name in zip(range(self.start_j, 0), _level_names(self.df.index)):
            tr.append(
                {
                    "j": j,
                    "i": i,
                    "rowspan": rowspan,
                    "value": level_name,
                    "elem": "th",
                    "scope": "column",
                    "role": "index-level-name",
                }
            )
        if has_col_levels:
            # make room for the headers that contain index level names by
            # lowering the row span of the cell above from 2 to 1
            thead_rows[-2][0]["rowspan"] = 1
            # fill the last thead row with the index level names
            thead_rows[-1] = tr
        else:
            # insert at the beginning of the very first table row
            thead_rows[0] = tr + thead_rows[0]
            # (the last thead row remains empty)

    def add_df_slices(self):
        # add the dataframe data, ie the top and bottom dataframe slices
        self.add_table_body(self.df.iloc[: self.top_slice_size], "top_slice", 0)
        if self.is_elided:
            self.add_ellipsis()
        if self.bottom_slice_size:
            self.add_table_body(
                self.df.iloc[-self.bottom_slice_size :],
                "bottom_slice",
                self.top_slice_size,
            )

    def add_table_body(self, sub_df, part_name, start_i):
        tbody = {"name": part_name, "elem": "tbody", "rows": []}
        self.parts.append(tbody)

        # the th that will be at the start of each row, containing the
        # dataframe index data
        table_headers = _multi_index_table_headers(sub_df.index, orientation="vertical")
        for tr_count, (table_row_headers, df_row) in enumerate(
            zip(table_headers, sub_df.itertuples(index=False, name=None))
        ):
            i = start_i + tr_count
            tr = []
            for j, th in zip(range(self.start_j, 0), table_row_headers):
                if th is not None:
                    th["j"] = j
                    th["i"] = i
                    th["elem"] = "th"
                    th["scope"] = "row"
                    th["role"] = "index-level-value"
                    tr.append(th)
            # after the table headers goes the actual dataframe data
            tr += [
                {
                    "value": v,
                    "i": i,
                    "j": j,
                    "elem": "td",
                    "column_idx": j,
                    "role": "dataframe-data",
                }
                for j, v in enumerate(df_row)
            ]
            tbody["rows"].append(tr)

    def add_ellipsis(self):
        self.parts.append({"name": "ellipsis", "elem": "tbody"})
