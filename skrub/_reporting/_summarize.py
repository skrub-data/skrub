"""Get information and plots for a dataframe, that are used to generate reports."""

import sys

from .. import _column_associations, _config
from .. import _dataframe as sbd
from . import _plotting, _sample_table, _utils

try:
    import pyarrow  # noqa: F401

    _PYARROW_INSTALLED = True
except ImportError:
    _PYARROW_INSTALLED = False


_SUBSAMPLE_SIZE = 3000
_N_TOP_ASSOCIATIONS = 20


def summarize_dataframe(
    df,
    *,
    order_by=None,
    with_plots=True,
    with_associations=True,
    title=None,
    max_top_slice_size=5,
    max_bottom_slice_size=5,
    verbose=1,
):
    """Collect information about a dataframe, used to produce reports.

    Parameters
    ----------
    df : dataframe
        The dataframe about which the summary/report needs to be generated.

    order_by : str or None, default=None
        The name of the column on which the dataframe should be sorted. If
        provided, it must correspond to a numeric or datetime column. Other
        numeric columns are plotted as a function of the sorting columns,
        rather than histograms.

    with_plots : bool, default=True
        Generate the images or not.

    with_associations : bool, default=True
        Compute the associations or not.

    title : str or None, default=None
        A title that gets added to the returned dictionary and can be picked up
        and inserted in the report.

    max_top_slice_size : int, default=5
        Maximum number of rows from the top of the dataframe to show in the
        sample table.

    max_bottom_slice_size : int, default=5
        Maximum number of rows from the end of the dataframe to show in the
        sample table.

    verbose : int, default = 1
        Whether to print progress information while the report is being generated.

        * verbose = 1 prints how many columns have been processed so far.
        * verbose = 0 silences the output.

    Returns
    -------
    dict
        A dictionary containing the extracted information.
    """
    n_rows, n_columns = sbd.shape(df)
    summary = {
        "dataframe": df,
        "dataframe_module": sbd.dataframe_module_name(df),
        "n_rows": n_rows,
        "n_columns": n_columns,
        "columns": [],
        "dataframe_is_empty": not n_rows or not n_columns,
        "plots_skipped": not with_plots,
        "associations_skipped": not with_associations,
        "sample_table": _sample_table.make_table(
            df,
            max_top_slice_size=max_top_slice_size,
            max_bottom_slice_size=max_bottom_slice_size,
        ),
    }
    if title is not None:
        summary["title"] = title
    if order_by is not None:
        df = sbd.sort(df, by=order_by)
        summary["order_by"] = order_by
    if order_by is None:
        order_by_column = None
    else:
        order_by_idx = sbd.column_names(df).index(order_by)
        order_by_column = sbd.col_by_idx(df, order_by_idx)
    for position in range(sbd.shape(df)[1]):
        if verbose > 0:
            print(
                f"Processing column {position + 1: >3} / {n_columns}",
                file=sys.stderr,
                end="\r",
                flush=True,
            )
        summary["columns"].append(
            _summarize_column(
                sbd.col_by_idx(df, position),
                position,
                dataframe_summary=summary,
                with_plots=with_plots,
                order_by_column=order_by_column,
            )
        )
    if verbose > 0:
        print(flush=True, file=sys.stderr)

    summary["n_constant_columns"] = sum(
        c["value_is_constant"] for c in summary["columns"]
    )
    if not _PYARROW_INSTALLED and summary["dataframe_module"] == "polars":
        with_associations = False
        summary["associations_skipped_polars_no_pyarrow"] = True
    elif with_associations:
        if n_rows and n_columns:
            _add_associations(df, summary)
        else:
            summary["top_associations"] = []
    return summary


def _add_associations(df, dataframe_summary):
    df = sbd.sample(df, n=min(sbd.shape(df)[0], _SUBSAMPLE_SIZE))
    associations = _column_associations.column_associations(df)

    # get only the top _N_TOP_ASSOCIATIONS
    associations = sbd.slice(associations, _N_TOP_ASSOCIATIONS)

    # transform dataframe into the format expected by the HTML template
    # (list of dicts):
    asso_dict = _utils.to_dict(associations)
    values = zip(*asso_dict.values())
    keys = list(asso_dict.keys())
    asso_list = [dict(zip(keys, vals)) for vals in values]

    dataframe_summary["top_associations"] = asso_list


def _summarize_column(
    column, position, dataframe_summary, *, with_plots, order_by_column
):
    summary = {
        "position": position,
        "idx": position,
        "name": sbd.name(column),
        "dtype": _utils.get_dtype_name(column),
        "value_is_constant": False,
        "is_ordered": False,
    }
    _add_nulls_summary(summary, column, dataframe_summary=dataframe_summary)
    if summary["null_count"] == dataframe_summary["n_rows"]:
        summary["plot_names"] = []
        return summary
    try:
        summary["n_unique"] = sbd.n_unique(column)
        summary["unique_proportion"] = summary["n_unique"] / max(
            1, dataframe_summary["n_rows"]
        )
        summary["is_high_cardinality"] = (
            summary["n_unique"] > _config.get_config()["cardinality_threshold"]
        )
    except Exception:
        # for some dtypes n_unique can fail eg with a typeerror for
        # non-hashable types in pandas.
        pass
    _add_value_counts(
        summary, column, dataframe_summary=dataframe_summary, with_plots=with_plots
    )
    _add_numeric_summary(
        summary,
        column,
        dataframe_summary=dataframe_summary,
        with_plots=with_plots,
        order_by_column=order_by_column,
    )
    _add_datetime_summary(summary, column, with_plots=with_plots)
    summary["plot_names"] = [k for k in summary.keys() if k.endswith("_plot")]
    _add_is_sorted(summary, column)

    return summary


def _add_nulls_summary(summary, column, dataframe_summary):
    null_count = sbd.sum(sbd.is_null(column))
    summary["null_count"] = null_count
    null_proportion = null_count / max(1, dataframe_summary["n_rows"])
    summary["null_proportion"] = null_proportion
    if summary["null_proportion"] == 0.0:
        summary["nulls_level"] = "ok"
    elif summary["null_proportion"] == 1.0:
        summary["nulls_level"] = "critical"
    else:
        summary["nulls_level"] = "warning"


def _add_value_counts(summary, column, *, dataframe_summary, with_plots):
    if sbd.is_numeric(column) or sbd.is_any_date(column) or sbd.is_duration(column):
        return
    n_unique, value_counts = _utils.top_k_value_counts(column, k=10)
    # if the column contains all nulls, _add_value_counts does not get called
    assert n_unique > 0

    # value_counts may be able to find the number of unique values in cases
    # where n_unique() fails (eg non-hashable column content in pandas) so we
    # update n_unique and unique_proportion
    summary["n_unique"] = n_unique
    summary["unique_proportion"] = n_unique / max(1, dataframe_summary["n_rows"])

    summary["value_counts"] = value_counts
    summary["most_frequent_values"] = [v for v, _ in value_counts]

    if n_unique == 1:
        summary["value_is_constant"] = True
        summary["constant_value"] = value_counts[0][0]
    else:
        summary["value_is_constant"] = False
        if with_plots:
            summary["value_counts_plot"] = _plotting.value_counts(
                value_counts,
                n_unique,
                dataframe_summary["n_rows"],
                color=_plotting.COLORS[1],
            )


def _add_datetime_summary(summary, column, with_plots):
    if not sbd.is_any_date(column):
        return
    min_date = sbd.min(column)
    max_date = sbd.max(column)
    if min_date == max_date:
        summary["value_is_constant"] = True
        summary["constant_value"] = min_date.isoformat()
        return
    summary["value_is_constant"] = False
    summary["min"] = min_date.isoformat()
    summary["max"] = max_date.isoformat()
    if with_plots:
        (
            summary["histogram_plot"],
            summary["n_low_outliers"],
            summary["n_high_outliers"],
        ) = _plotting.histogram(column, color=_plotting.COLORS[0])


def _add_numeric_summary(
    summary, column, dataframe_summary, with_plots, order_by_column
):
    del dataframe_summary
    assert len(column)  # _add_numeric_summary is not called if the column is empty
    first_value = sbd.to_list(sbd.head(column, 1))[0]
    if sbd.is_duration(column):
        summary["is_duration"] = True
        column, duration_unit = _utils.duration_to_numeric(column)
    else:
        summary["is_duration"] = False
        if not sbd.is_numeric(column):
            if sbd.is_bool(column):
                summary["mean"] = sbd.mean(column)
            return
        duration_unit = None
    summary["duration_unit"] = duration_unit
    std = sbd.std(column)
    summary["standard_deviation"] = float("nan") if std is None else std
    summary["mean"] = sbd.mean(column)
    quantiles = _utils.quantiles(column)
    summary["inter_quartile_range"] = quantiles[0.75] - quantiles[0.25]
    if quantiles[0.0] == quantiles[1.0]:
        summary["value_is_constant"] = True
        summary["constant_value"] = first_value
        return
    summary["value_is_constant"] = False
    summary["quantiles"] = quantiles
    if not with_plots:
        return
    if order_by_column is None:
        (
            summary["histogram_plot"],
            summary["n_low_outliers"],
            summary["n_high_outliers"],
        ) = _plotting.histogram(
            column, duration_unit=duration_unit, color=_plotting.COLORS[0]
        )
    else:
        summary["line_plot"] = _plotting.line(order_by_column, column)


def _add_is_sorted(summary, column):
    summary["is_ordered"] = sbd.is_sorted(column, descending=False) or sbd.is_sorted(
        column, descending=True
    )
