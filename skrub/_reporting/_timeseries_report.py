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

        self.time = SelectCols(s.any_date()).fit_transform(df)
        self.time_range = (self.time[:, 0].min(), self.time[:, 0].max())
        self.ycols = SelectCols(s.all() - s.any_date()).fit_transform(df)
        self.ncols = 3
        self.nrows = (self.ycols.width + self.ncols - 1) // self.ncols
        self.colors = qualitative.Plotly

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

                # Use colors 7 and 8 for overview (avoid colors 0-6 used in metadata)
                overview_colors = qualitative.Plotly
                fig = self.make_overview(
                    y_col, color_a=overview_colors[7], color_b=overview_colors[8]
                )
                metadata = self.get_metadata(y_col)

                plots = {
                    "overview": fig,
                    "trends": self.plot_avg(y_col, y_min=y_min, y_max=y_max),
                    #              "extrema": self.make_extrema(y_col),
                    "autocorrelation": fig,  # self.plot_autocorrelation(y_col),
                    # 'periodicity': self.plot_avg(y_col),
                    #    "correlations": fig,
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
            x = self.time[:, 0].to_numpy().ravel()
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

        # Calculate y_min and y_max once for consistent ranges across all plots
        y_min = y.min()
        y_max = y.max()

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
            y_min,
            y_max,
            y.null_count(),
            0,
        ]

        # Use qualitative colors for each metadata label
        colors = qualitative.Plotly

        plots = [
            self.make_mean_line(y, colors[0], y_min=y_min, y_max=y_max),
            self.make_horizontal_bands(
                y, colors[1], band_type="std", y_min=y_min, y_max=y_max
            ),
            self.make_horizontal_bands(
                y, colors[2], band_type="percentile", y_min=y_min, y_max=y_max
            ),
            self.make_extrema(y, overlay=True, color_a=colors[3], color_b=colors[4]),
            self.make_extrema(y, overlay=True, color_a=colors[3], color_b=colors[4]),
            self.make_extrema(y, overlay=True, color_a=colors[5], color_b=colors[5]),
            self.make_extrema(y, overlay=True, color_a=colors[6], color_b=colors[6]),
        ]

        # Create base plot with just the bottom timeline (no rectangles)
        base_plot_html = self.make_plot(
            y, (0, 1), (0, 1), ("", "", ""), bottom=True, y_min=y_min, y_max=y_max
        )

        return {
            "base_plot": base_plot_html,
            "items": {
                label: {
                    "value": f"{values[i]:.2f}",
                    "plot": plots[i],
                    "color": colors[i % len(colors)],
                    "has_button": label in labels,
                }
                for i, label in enumerate(labels)
            },
        }

    def make_mean_line(self, y, color, y_min=None, y_max=None):
        """Create transparent overlay plot with just a horizontal line at the mean."""
        if y_min is None:
            y_min = y.min()
        if y_max is None:
            y_max = y.max()
        mean_value = y.mean()
        print(f"{y.name} mean = {mean_value}")
        time_min = self.time[:, 0].min()
        time_max = self.time[:, 0].max()

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{}, {}],  # row 1: two separate plots
                [{"colspan": 2}, None],  # row 2: one plot spanning both columns
            ],
            row_heights=[0.75, 0.25],
            subplot_titles=("", "", ""),
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
        )

        # Top two plots - empty and invisible
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                showlegend=False,
                visible=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                showlegend=False,
                visible=False,
            ),
            row=1,
            col=2,
        )

        # Bottom plot - just a horizontal line at the mean
        fig.add_trace(
            go.Scatter(
                x=[time_min, time_max],
                y=[mean_value, mean_value],
                mode="lines",
                showlegend=False,
                line=dict(color=color, width=2),
            ),
            row=2,
            col=1,
        )

        # Hide top subplot axes completely
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

        # Match x-axis range to base plot
        time_range = time_max - time_min
        time_padding = time_range * 0.01
        fig.update_xaxes(
            range=[time_min - time_padding, time_max + time_padding],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=2,
            col=1,
        )

        # Hide y-axis but keep the trace visible, match range to data
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        fig.update_yaxes(
            range=[y_min - y_padding, y_max + y_padding],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=2,
            col=1,
        )

        # Transparent background
        fig.update_layout(
            title_text="",
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig.to_html(full_html=False, config={"responsive": True})

    def make_horizontal_bands(self, y, color, band_type="std", y_min=None, y_max=None):
        """Create transparent overlay with horizontal bands.

        band_type: 'std' for mean ± std, 'percentile' for 25th/75th percentiles
        """
        if y_min is None:
            y_min = y.min()
        if y_max is None:
            y_max = y.max()
        mean_value = y.mean()
        time_min = self.time[:, 0].min()
        time_max = self.time[:, 0].max()

        if band_type == "std":
            offset = y.std()
            lower = mean_value - offset
            upper = mean_value + offset
        else:  # percentile
            lower = y.quantile(0.25)
            upper = y.quantile(0.75)

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            row_heights=[0.75, 0.25],
            subplot_titles=("", "", ""),
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
        )

        # Top two plots - empty and invisible
        for col in [1, 2]:
            fig.add_trace(
                go.Scatter(x=[], y=[], mode="lines", showlegend=False, visible=False),
                row=1,
                col=col,
            )

        # Bottom plot - transparent trace
        fig.add_trace(
            go.Scatter(
                x=[time_min],
                y=[mean_value],
                mode="markers",
                showlegend=False,
                marker=dict(size=0),
            ),
            row=2,
            col=1,
        )

        # Add horizontal bands
        for y0, y1 in [(lower, mean_value), (mean_value, upper)]:
            fig.add_hrect(
                y0=y0,
                y1=y1,
                fillcolor=color,
                opacity=0.3,
                line_width=0,
                row=2,
                col=1,
            )

        # Hide all axes
        for col in [1, 2]:
            fig.update_xaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=1,
                col=col,
            )
            fig.update_yaxes(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=1,
                col=col,
            )

        # Match x and y axis ranges to base plot
        time_range = time_max - time_min
        time_padding = time_range * 0.01
        fig.update_xaxes(
            range=[time_min - time_padding, time_max + time_padding],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=2,
            col=1,
        )

        # Match y-axis range to data with padding
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        fig.update_yaxes(
            range=[y_min - y_padding, y_max + y_padding],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=2,
            col=1,
        )

        fig.update_layout(
            title_text="",
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig.to_html(full_html=False, config={"responsive": True})

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

        # Top two plots - invisible if bottom mode
        fig.add_trace(
            go.Scatter(
                x=self.time[:, 0][slice(*range1)],
                y=y[slice(*range1)],
                mode="lines+markers",
                showlegend=False,
                visible=not bottom,
                opacity=1.0,
                line=dict(width=2, color=color_a),
                marker=dict(size=4, color=color_a),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.time[:, 0][slice(*range2)],
                y=y[slice(*range2)],
                mode="lines+markers",
                showlegend=False,
                visible=not bottom,
                opacity=1.0,
                line=dict(width=2, color=color_b),
                marker=dict(size=4, color=color_b),
            ),
            row=1,
            col=2,
        )

        # Bottom plot trace
        # Always visible (needed for vrects and base-plot sparkline)
        # Transparent in overlay mode so base-plot shows through
        fig.add_trace(
            go.Scatter(
                x=self.time[:, 0],
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
                x0=min(self.time[:, 0][slice(*range1)]),
                x1=max(self.time[:, 0][slice(*range1)]),
                fillcolor=color_a,
                opacity=0.2,
                line_width=0,
                row=2,
                col=1,
            )
            fig.add_vrect(
                x0=min(self.time[:, 0][slice(*range2)]),
                x1=max(self.time[:, 0][slice(*range2)]),
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
        time_min = self.time[:, 0].min()
        time_max = self.time[:, 0].max()
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
            go.Scatter(
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

    def plot_avg(self, y, y_min=None, y_max=None):
        if y_min is None:
            y_min = y.min()
        if y_max is None:
            y_max = y.max()

        resolutions = [7, 14, 30, 90]
        time_values = self.time[:, 0].to_numpy()

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
                go.Scatter(
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
            go.Scatter(
                x=self.time[:, 0],
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
                go.Scatter(
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
        time_min = self.time[:, 0].min()
        time_max = self.time[:, 0].max()
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

        return fig.to_html(full_html=False, config={"responsive": True})

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

        return fig.to_html(full_html=False, config={"responsive": True})

    def stats_overview(self, df):
        time_col = self.time.columns[0]
        deltas = self.time[time_col].diff()
        sampling_period = deltas.mode().item().total_seconds()
        sampling_freq = 1.0 / sampling_period
        start_date = self.time[time_col].min()
        end_date = self.time[time_col].max()
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
