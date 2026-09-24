import argparse
from pathlib import Path

import jinja2
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots

from .. import Cleaner, SelectCols
from .. import selectors as s


def get_env():
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            Path(__file__).resolve().parent / "_data" / "templates",
            encoding="UTF-8",
        ),
        autoescape=True,
    )


def format_column_name(name):
    """Format column name: add line break at first space after 10 chars if name is >10
    chars and has space."""
    if len(name) > 10 and " " in name:
        # Find first space after character 10
        for i in range(10, len(name)):
            if name[i] == " ":
                return name[:i] + "<br>" + name[i + 1 :]
    return name


def calculate_label_width(name, char_width_px=10):
    """Calculate approximate width needed for a formatted column name.

    Uses character width estimation. For names with line breaks,
    returns width of the longest line. Uses 10px per char to account
    for bold font weight.
    """
    if "<br>" in name:
        lines = name.split("<br>")
        return max(len(line) for line in lines) * char_width_px
    return len(name) * char_width_px


class TimeSeriesReport:
    def __init__(
        self,
        df,
        output_file,
        target=None,
        numcols=None,
        catcols=None,
        debug=False,
    ):
        self.df = df
        self.output_file = output_file

        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        df = Cleaner().fit_transform(df)
        time = SelectCols(s.any_date()).fit_transform(df)
        self.df = df.sort(time.columns)
        self.target = target
        self.time = SelectCols(s.any_date()).fit_transform(df)[:, 0]
        self.time_min = self.time.min()
        self.time_max = self.time.max()
        self.time_range = (self.time_min, self.time_max)

        if numcols == "all":
            self.numcols = SelectCols(s.numeric()).fit_transform(self.df)
        else:
            self.numcols = numcols
        if catcols == "all":
            self.catcols = SelectCols(s.string() & s.categorical()).fit_transform(
                self.df
            )
        elif catcols == "string":
            self.catcols = SelectCols(s.string()).fit_transform(self.df)
        elif catcols == "categorical":
            self.catcols = SelectCols(s.categorical()).fit_transform(self.df)
        else:
            self.catcols = catcols

        self.ycols = SelectCols(s.all() - s.any_date()).fit_transform(df)
        self.ncols = 3
        self.nrows = (self.ycols.width + self.ncols - 1) // self.ncols
        self.colors = qualitative.Plotly
        self.scatter_kwargs = {
            "mode": "lines+markers",
            "showlegend": False,
            "visible": False,
        }

        if debug:
            pass

        else:
            figs_overview, max_label_width = self.plot_overview()

            overview = {
                "title": "Dataset Overview",
                "caption": "This is a summary of the dataset.",
                "figs": figs_overview,
                "stats": self.stats_overview(self.ycols),
                "max_label_width": max_label_width + 40,  # Add padding for button
            }

            template_data = {"overview": overview}

            print(template_data)

            # Per-column individual plots for column tabs
            columns = []
            for i, col in enumerate(self.ycols.columns[:10]):
                y_col = self.ycols[col]
                # Calculate y_min and y_max once per column for consistent ranges
                y_min = y_col.min()
                y_max = y_col.max()

                # Create the main figure with all toggleable traces
                fig = self._make_figure(y_col, "test")

                # Get metadata items (values, colors, etc.)
                metadata = self.get_metadata(y_col)

                # MEMORY OPTIMIZATION:
                # Load Plotly.js only once (first plot), use unique div IDs
                col_id = col.replace(" ", "-").replace("_", "-")
                include_plotlyjs = "cdn" if i == 0 else False
                plotly_html = fig.to_html(
                    full_html=False,
                    include_plotlyjs=include_plotlyjs,
                    div_id=f"plot-metadata-{col_id}",
                    config={"responsive": True},
                )
                metadata["plot"] = plotly_html

                # SIZE COMPARISON (only for first column to avoid spam)
                if i == 0:
                    print(f"\n=== HTML Size Comparison for '{col}' ===")

                    # New approach: Single Plotly HTML
                    plotly_size = len(plotly_html.encode("utf-8"))

                    # Old approach: Estimate 4 SVG embeds
                    # Typical matplotlib SVG is ~10-15KB, but varies by complexity
                    estimated_svg_size_each = 12 * 1024  # 12KB per SVG
                    estimated_total_svg = estimated_svg_size_each * 4

                    print(
                        f"New (1 Plotly HTML): {plotly_size:,} bytes "
                        f"({plotly_size / 1024:.1f} KB)"
                    )
                    print(
                        f"Old (4 SVG embeds, est.): {estimated_total_svg:,} "
                        f"bytes ({estimated_total_svg / 1024:.1f} KB)"
                    )
                    print(
                        f"Difference: {plotly_size - estimated_total_svg:+,} "
                        f"bytes ({(plotly_size - estimated_total_svg) / 1024:+.1f} KB)"
                    )
                    print(
                        f"Ratio: Plotly is ~{plotly_size / estimated_total_svg:.1f}x "
                        f"estimated SVG size"
                    )
                    print(
                        "Note: Plotly includes interactivity (zoom, pan, toggle traces)"
                    )
                    print("=" * 50 + "\n")

                # Other plot views
                plots = {
                    "trends": self.plot_avg(
                        y_col,
                        y_min=y_min,
                        y_max=y_max,
                        div_id=f"plot-trends-{col_id}",
                        include_plotlyjs=False,  # Already loaded in metadata plot
                    ),
                    #              "extrema": self.make_extrema(y_col),
                    # "autocorrelation": ...,
                    # 'periodicity': self.plot_avg(y_col),
                }
                columns.append({"name": col, "plots": plots, "metadata": metadata})
            template_data["columns"] = columns

            template = get_env().get_template("timeseries_report.html")
            html = template.render(template_data)
            Path(output_file).write_text(html, "UTF-8")

    def plot_overview(self, ncols=3):
        import io
        import re

        figs = []
        max_width = 0

        for i, col in enumerate(self.ycols[:, :10]):
            x = self.time.to_numpy().ravel()
            y = col.to_numpy().ravel()

            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(x, y, color=self.colors[i % len(self.colors)])
            ax.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            buf = io.BytesIO()
            fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            svg = buf.getvalue().decode("utf-8")
            # Rebuild the <svg> tag: keep viewBox
            # override width/height to fill container
            start = svg.find("<svg ")
            end = svg.find(">", start)
            old_tag = svg[start : end + 1]
            viewbox_match = re.search(r'viewBox="[^"]*"', old_tag)
            viewbox = viewbox_match.group(0) if viewbox_match else ""
            new_tag = (
                f'<svg width="100%" height="100%" {viewbox} preserveAspectRatio="none">'
            )
            svg = svg[:start] + new_tag + svg[end + 1 :]

            # Format column name and track max width
            formatted_name = format_column_name(col.name)
            width = calculate_label_width(formatted_name)
            max_width = max(max_width, width)

            figs.append((col.name, formatted_name, col.dtype, svg))

        return figs, max_width

    def make_overview(self, y, color_a="#1f77b4", color_b="#d62728"):
        y_min = y.min()
        y_max = y.max()
        if len(y) > 3e4:
            nsamples = 10000
        else:
            nsamples = len(y) // 10
        titles = f"First {nsamples} Points", f"Last {nsamples} Points", ""
        range1 = (0, nsamples)
        range2 = (len(y) - nsamples, len(y))

        return self.make_plot(
            y,
            range1,
            range2,
            titles,
            overlay=True,
            color_a=color_a,
            color_b=color_b,
            y_min=y_min,
            y_max=y_max,
        )

    def make_extrema(self, y, overlay=True, color_a="#1f77b4", color_b="#d62728"):
        y_min = y.min()
        y_max = y.max()
        min_idx = y.arg_min()
        max_idx = y.arg_max()
        titles = "Minimum", "Maximum", ""
        # Clamp slice bounds to avoid negative start indices, which can
        # produce empty slices in Polars and cause a "min() arg is empty" error.
        if len(y) > 3e4:
            half_window = 10000
        else:
            half_window = len(y) // 10
        range1 = (max(0, min_idx - half_window), min_idx + half_window)
        range2 = (max(0, max_idx - half_window), max_idx + half_window)

        return self.make_plot(
            y,
            range1,
            range2,
            titles,
            overlay=overlay,
            color_a=color_a,
            color_b=color_b,
            y_min=y_min,
            y_max=y_max,
        )

    def find_times(self):
        pass

    def get_metadata(self, y):
        # data type
        # min, max
        # constant period
        # percentiles
        # periods of missing data
        # autocorrelation
        # periodicity
        # outliers?
        # sampling rate
        # data range

        labels = [
            "mean",
            "std",
            "percentile",
            "min",
            "max",
            "missingness",
            "outliers",
        ]

        values = [
            y.mean(),
            y.std(),
            y.quantile([0.25, 0.5, 0.75])[0],
            y.min(),
            y.max(),
            y.null_count(),
            0,
        ]

        # Use qualitative colors for each metadata label
        colors = qualitative.Plotly

        return {
            "items": {
                label: {
                    "value": f"{values[i]:.2f}",
                    "color": colors[i % len(colors)],
                    "has_button": label in labels,
                }
                for i, label in enumerate(labels)
            },
        }

    def _add_mean(self, y, color, name="mean"):
        mean = y.mean()
        return go.Scattergl(
            x=[self.time_min, self.time_max],
            y=[mean, mean],
            name=name,
            line=dict(color=color, width=2),
            mode="lines",
            visible=False,
            showlegend=False,
        )

    def _add_std(self, y, color, name="std"):
        lower = y.mean() - y.std()
        upper = y.mean() + y.std()
        band_x = [
            self.time_min,
            self.time_max,
            self.time_max,
            self.time_min,
            self.time_min,
        ]
        band_y = [lower, lower, upper, upper, lower]

        return self._add_bands([band_x, band_y], color, name)

    def _add_percentile(self, y, color, quantile=0.25, name="percentile"):
        lower = y.quantile(quantile)
        upper = y.quantile(quantile + 0.50)
        band_x = [
            self.time_min,
            self.time_max,
            self.time_max,
            self.time_min,
            self.time_min,
        ]
        band_y = [lower, lower, upper, upper, lower]

        return self._add_bands([band_x, band_y], color, name)

    def _add_bands(self, band_corners, color, name):
        band_x, band_y = band_corners

        return go.Scattergl(
            x=band_x,
            y=band_y,
            name=name,
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            opacity=0.3,
            mode="lines",
            visible=False,
            showlegend=False,
        )

    def _add_timeline(self, y):
        return go.Scattergl(
            x=self.time,
            y=y,
            name="timeline",
            line=dict(color="black", width=1),
            mode="lines",
            visible=True,
            showlegend=False,
        )

    def _make_trace(self, x, y, name, visible=False, color="#1f77b4"):
        return go.Scattergl(
            x=x,
            y=y,
            name=name,
            mode="lines+markers",
            visible=visible,
            showlegend=False,
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color),
        )

    def _make_figure(
        self,
        y,
        labels,
        color="#1f77b4",
    ):
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{}, {}],  # row 1: two separate plots
                [{"colspan": 2}, None],  # row 2: one plot spanning both columns
            ],
            row_heights=[0.75, 0.25],
            subplot_titles=("", "", ""),  # Titles set dynamically by JavaScript
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
        )
        heads = self._make_trace(self.time, y, "heads", color="red", visible=True)
        tails = self._make_trace(self.time, y, "tails", color="red", visible=True)
        min = self._make_trace(self.time, y, "min", color="yellow")
        max = self._make_trace(self.time, y, "max", color="yellow")
        missingness = self._make_trace(self.time, y, "missingness", color="purple")
        outliers = self._make_trace(self.time, y, "outliers", color="purple")
        timeline = self._add_timeline(y)
        mean = self._add_mean(y, "green")
        std = self._add_std(y, "orange")
        percentile = self._add_percentile(y, "blue")

        fig.add_trace(heads, row=1, col=1)
        fig.add_trace(tails, row=1, col=2)
        fig.add_trace(mean, row=2, col=1)
        fig.add_trace(std, row=2, col=1)
        fig.add_trace(percentile, row=2, col=1)
        fig.add_trace(min, row=1, col=1)
        fig.add_trace(max, row=1, col=2)
        fig.add_trace(missingness, row=1, col=1)
        fig.add_trace(outliers, row=1, col=2)
        fig.add_trace(timeline, row=2, col=1)

        return fig

    def make_plot(
        self,
        y,
        range1,
        range2,
        titles,
        overlay=False,
        bottom=False,
        base_plot=True,
        color_a="#1f77b4",
        color_b="#d62728",
        y_min=None,
        y_max=None,
    ):
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{}, {}],  # row 1: two separate plots
                [{"colspan": 2}, None],  # row 2: one plot spanning both columns
            ],
            row_heights=[0.75, 0.25],
            subplot_titles=(titles),
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
        )

        # Bottom plot trace
        # Always visible (needed for vrects and base-plot sparkline)
        # Transparent in overlay mode so base-plot shows through
        fig.add_trace(
            go.Scattergl(
                x=self.time,
                y=y,
                mode="lines",
                showlegend=False,
                visible=True,
                line=dict(color="rgba(0,0,0,0)" if overlay else "black", width=1),
            ),
            row=2,
            col=1,
        )

        # Rectangles visible except in bottom-only mode
        if not bottom:
            fig.add_vrect(
                x0=min(self.time[slice(*range1)]),
                x1=max(self.time[slice(*range1)]),
                fillcolor=color_a,
                opacity=0.2,
                line_width=0,
                row=2,
                col=1,
            )
            fig.add_vrect(
                x0=min(self.time[slice(*range2)]),
                x1=max(self.time[slice(*range2)]),
                fillcolor=color_b,
                opacity=0.2,
                line_width=0,
                row=2,
                col=1,
            )

        # Set y-axis range for first two subplots
        # Use provided y_min/y_max or calculate from data
        if y_min is None:
            y_min = y.min()
        if y_max is None:
            y_max = y.max()
        y_range = y_max - y_min
        padding = y_range * 0.05  # 5% padding
        y_min_padded = y_min - padding
        y_max_padded = y_max + padding
        fig.update_yaxes(range=[y_min_padded, y_max_padded], row=1, col=1)
        fig.update_yaxes(range=[y_min_padded, y_max_padded], row=1, col=2)

        # Add padding to x-axis of bottom plot
        time_min = self.time.min()
        time_max = self.time.max()
        time_range = time_max - time_min
        time_padding = time_range * 0.01
        fig.update_xaxes(
            range=[time_min - time_padding, time_max + time_padding], row=2, col=1
        )

        # Hide or completely remove top two subplots
        if bottom:
            # Completely hide top subplot axes
            fig.update_xaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=1,
                col=1,
            )
            fig.update_yaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=1,
                col=1,
            )
            fig.update_xaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=1,
                col=2,
            )
            fig.update_yaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=1,
                col=2,
            )
            # Hide subplot titles for top plots
            if len(fig.layout.annotations) >= 2:
                fig.layout.annotations[0].update(text="")
                fig.layout.annotations[1].update(text="")
        else:
            # Just hide tick labels for overlay/normal mode
            fig.update_xaxes(showticklabels=False, row=1, col=1)
            fig.update_yaxes(showticklabels=False, row=1, col=1)
            fig.update_xaxes(showticklabels=False, row=1, col=2)
            fig.update_yaxes(showticklabels=False, row=1, col=2)

        # Make bottom subplot transparent in overlay mode
        # (keep axes visible so vrects render, but hide all decorations)
        if overlay:
            fig.update_xaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                row=2,
                col=1,
            )
            fig.update_yaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                row=2,
                col=1,
            )

        # Layout settings
        layout_config = {
            "title_text": "",
            "autosize": True,
            "margin": dict(l=40, r=40, t=40, b=40),
        }

        # Transparent background for overlay mode only
        if overlay:
            layout_config["plot_bgcolor"] = "rgba(0,0,0,0)"
            layout_config["paper_bgcolor"] = "rgba(0,0,0,0)"

        fig.update_layout(**layout_config)

        return fig.to_html(full_html=False, config={"responsive": True})

    def agg_by_period(self, period, df, datetime_col="start", by_year=False):
        base = (
            df.group_by(pl.col(datetime_col).dt.truncate(period).alias("period"))
            .agg(pl.len().alias("count"))
            .sort("period")
        )

        if not by_year:
            return base.select("period", "count")

        return (
            base.with_columns(pl.col("period").dt.year().alias("year"))
            .select("year", "period", "count")
            .sort("year", "period")
        )

    def plot_agg(self, df):
        df_1h = self.agg_by_period("1h", df)

        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=df_1h["period"],
                y=df_1h["count"],
                name="1h",
                mode="lines",
                line=dict(width=1),
            )
        )

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                ),
                rangeslider=dict(visible=True),
                type="date",
            ),
            # title="Bixi hourly trips (1h buckets)",
            # yaxis_title="Trips per hour"
        )

        return fig.to_html(full_html=False, config={"responsive": True})

    def plot_avg(self, y, y_min=None, y_max=None, div_id=None, include_plotlyjs=False):
        if y_min is None:
            y_min = y.min()
        if y_max is None:
            y_max = y.max()

        resolutions = [7, 14, 30, 90]
        time_values = self.time.to_numpy()

        # Create 2-row layout: top for windowed averages, bottom for timeline
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.75, 0.25],
            subplot_titles=["Windowed Averages", ""],
            vertical_spacing=0.15,
        )

        # --- Build one trace per window in top subplot ---
        values = pd.Series(y)
        # Get default Plotly colors
        default_colors = qualitative.Plotly

        for i, w in enumerate(resolutions):
            y_rolled = values.rolling(w, min_periods=1).mean().values
            name = f"{w}-day avg"
            color = default_colors[i % len(default_colors)]

            # Add to top subplot
            fig.add_trace(
                go.Scattergl(
                    x=time_values,
                    y=y_rolled,
                    mode="lines",
                    name=name,
                    showlegend=False,
                    visible=(i == 1),  # default: show 14-day
                    line=dict(width=2, color=color),
                ),
                row=1,
                col=1,
            )

        # --- Add bottom timeline sparkline (always visible) ---
        fig.add_trace(
            go.Scattergl(
                x=self.time,
                y=y,
                mode="lines",
                showlegend=False,
                line=dict(color="black", width=1),
            ),
            row=2,
            col=1,
        )

        # --- Add windowed averages as overlays on bottom plot ---
        for i, w in enumerate(resolutions):
            y_rolled = values.rolling(w, min_periods=1).mean().values
            color = default_colors[i % len(default_colors)]

            fig.add_trace(
                go.Scattergl(
                    x=time_values,
                    y=y_rolled,
                    mode="lines",
                    showlegend=False,
                    visible=(i == 1),  # default: show 14-day
                    line=dict(width=2, color=color),
                    opacity=0.75,
                ),
                row=2,
                col=1,
            )

        # --- Build update buttons ---
        buttons = []
        for i, w in enumerate(resolutions):
            label = f"{w}-day"
            # visibility list:
            # - first N are top windowed traces
            # - next 1 is bottom sparkline (always visible)
            # - last N are bottom windowed overlays
            visibility = (
                [j == i for j in range(len(resolutions))]  # top traces
                + [True]  # bottom sparkline always visible
                + [j == i for j in range(len(resolutions))]  # bottom overlays
            )

            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {"visible": visibility},
                    ],
                )
            )

        # Calculate y-axis range with padding
        y_range = y_max - y_min
        y_padding = y_range * 0.05

        # Calculate x-axis range with padding
        time_min = self.time.min()
        time_max = self.time.max()
        time_range = time_max - time_min
        time_padding = time_range * 0.01

        # Update axes for top subplot
        fig.update_xaxes(
            range=[time_min - time_padding, time_max + time_padding], row=1, col=1
        )
        fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding], row=1, col=1)

        # Update axes for bottom subplot - keep visible
        fig.update_xaxes(
            range=[time_min - time_padding, time_max + time_padding], row=2, col=1
        )
        fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding], row=2, col=1)

        fig.update_layout(
            title_text="",
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    buttons=buttons,
                    showactive=True,
                )
            ],
            hovermode="x unified",
        )

        return fig.to_html(
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            div_id=div_id,
            config={"responsive": True},
        )

        fig.update_layout(
            title="Time series — 7-day average",
            xaxis_title="Date",
            yaxis_title="Value",
            yaxis=dict(range=[y_min - y_padding, y_max + y_padding]),
            autosize=True,
            width=None,
            height=None,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    buttons=buttons,
                    showactive=True,
                )
            ],
            hovermode="x unified",
            template="plotly_white",
        )

        # figs = [f.to_html(full_html=False) for f in figs]

        return fig.to_html(
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            div_id=div_id,
            config={"responsive": True},
        )

    def stats_overview(self, df):
        deltas = self.time.diff()
        sampling_period = deltas.mode().item().total_seconds()
        sampling_freq = 1.0 / sampling_period
        start_date = self.time_min
        end_date = self.time_max
        period = end_date - start_date
        stats = {
            "sampling period": str(sampling_period),
            "sampling frequency": f"{sampling_freq:.6f} Hz",
            "date range": f"{start_date} to {end_date}",
            "period covered": f"{period} secs",
            #    "per_column stats": self.stats_numerical(self.numcols)
        }

        # for col in df.columns:
        #     stats[col.name] = {
        #         "mean": col.mean(),
        #         "std": col.std(),
        #         "min": col.min(),
        #         "25%": col.quantile(0.25),
        #         "50%": col.median(),
        #         "75%": col.quantile(0.75),
        #         "max": col.max(),
        #     }

        return stats

    def stats_numerical(self, df):
        stats = {}

        for col in self.numcols:
            stats[col.name] = {
                "mean": col.mean(),
                "std": col.std(),
                "min": col.min(),
                # "25%": col.quantile(0.25),
                # "50%": col.median(),
                # "75%": col.quantile(0.75),
                # "max": col.max(),
            }

        return stats

    def round_sampling_rate(self):
        pass

    def plot_categorical(self, df):
        fig = make_subplots(rows=1, cols=2)

        counts = (
            df.with_columns(pl.col("start").dt.truncate("1d").alias("time"))
            .group_by(["time", "start_arr"])
            .len()
            .pivot(on="start_arr", index="time", values="len")
            .fill_null(0)
            .sort("time")
        )
        counts_pd = counts.to_pandas().set_index("time")

        fig.add_trace(
            px.imshow(counts_pd.T, aspect="auto", color_continuous_scale="Blues").data[
                0
            ],
            row=1,
            col=1,
        )

        trips = (
            df.filter(pl.col("trip time") <= pl.duration(hours=3))["trip time"]
            .dt.total_minutes()
            .to_list()
        )

        fig.add_trace(go.Histogram(x=trips), row=1, col=2)

        return fig

    def computeACF(self, y, nlags=40):
        from statsmodels.tsa.stattools import acf

        acf_values = acf(y, nlags=nlags)
        return acf_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("-o", "--output_file", default="timeseries_report.html")
    args = parser.parse_args()
    input_file = Path(args.input_file)
    if input_file.suffix == ".parquet":
        df = pl.read_parquet(input_file)
    else:
        df = pl.read_csv(input_file)
    output_file = Path(args.output_file)
    output_file = "timeseries_report.html"
    TimeSeriesReport(df, output_file, target=None, numcols=None, catcols=None)
