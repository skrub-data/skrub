# ruff: noqa
from collections import Counter
from itertools import combinations

import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

TARGET_COL = "fraud_flag"
MAX_ITEMS = 24


def get_group_cols(column, max_items=MAX_ITEMS):
    """Create a list of column names"""
    return [f"{column}{idx}" for idx in range(1, max_items + 1)]


def plot_n_items_per_basket(df, n_most_freq=24):
    """Compare the distribution of nbr{i} between
    a fraud and a legit transaction.

    As most of the nbr{i} are 1, we remove them from the plot,
    to ease the observation of the values higher than 1.
    """
    fraud_items = _get_most_frequent_fraud_items(df, n_most_freq)
    df_n_items = _count_items_per_basket(df, fraud_items)

    fig, axes = plt.subplots(
        figsize=(8, 10),
        nrows=n_most_freq // 2,
        ncols=2,
    )

    for item, ax in zip(fraud_items, axes.flatten()):
        # Display the distribution as a bar chart.
        sns.barplot(
            df_n_items.query("item == @item and n_item > 1"),
            x="n_item",
            y="proportion",
            hue=TARGET_COL,
            ax=ax,
        )

        # Add total to the legend
        if not df_n_items.query("n_item > 1").empty:
            l = ax.legend()
            item_mask = df_n_items["item"] == item
            not_zero_mask = df_n_items["n_item"] > 0
            for idx, label in enumerate(l.get_texts()):
                target_mask = df_n_items[TARGET_COL] == idx
                total = df_n_items.loc[item_mask & target_mask & not_zero_mask][
                    "total"
                ].sum()
                label.set_text(f"{label.get_text()} (total: {total:,})")
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        # Add post-hoc labels to the bar
        if len(ax.containers) > 1:
            container = ax.containers[1]
            item_mask = df_n_items["item"] == item
            target_mask = df_n_items[TARGET_COL] == 1
            skip_first_mask = df_n_items["n_item"] > 1
            labels = df_n_items.loc[item_mask & target_mask & skip_first_mask][
                "total"
            ].values  # remove nbr = "1"
            ax.bar_label(container, labels)

        ax.yaxis.label.set_visible(False)
        ax.xaxis.label.set_visible(False)
        ax.set_title(item)

    plt.tight_layout()


def _get_most_frequent_fraud_items(df, n_most_freq):
    item_cols = get_group_cols("item")
    most_fraud_items = df.loc[df[TARGET_COL] == 1][item_cols].values.ravel()
    return pd.Series(most_fraud_items).value_counts().head(n_most_freq).index


def _count_items_per_basket(df, frequent_items):
    dfs = []
    for item in frequent_items:
        for target_value in [0, 1]:
            df_ = df.loc[df[TARGET_COL] == target_value]
            item_freq = (
                (df_ == item)
                .sum(axis=1)
                .value_counts(normalize=True)
                .sort_index()
                .to_frame()
                .reset_index()
            )
            item_freq.columns = ["n_item", "proportion"]
            item_counts = (
                (df_ == item)
                .sum(axis=1)
                .value_counts(normalize=False)
                .sort_index()
                .to_frame()
                .reset_index()
            )
            item_freq["total"] = item_counts["count"]
            item_freq[TARGET_COL] = target_value
            item_freq["item"] = item
            dfs.append(item_freq)

    return pd.concat(dfs)


def plot_nbr_per_item(df, n_most_freq=24):
    """Compare the distribution of nbr{i} between
    a fraud and a legit transaction.

    As most of the nbr{i} are 1, we remove them from the plot,
    to ease the observation of the values higher than 1.
    """
    fraud_items = _get_most_frequent_fraud_items(df, n_most_freq)
    df = _melt_dataframe(df, group_col1="item", group_col2="nbr")

    fig, axes = plt.subplots(
        figsize=(8, 10),
        nrows=n_most_freq // 2,
        ncols=2,
    )

    for item, ax in zip(fraud_items, axes.flatten()):
        item_mask = df["item"] == item
        nbrs, totals = [], []

        for target_value in [0, 1]:
            target_mask = df[TARGET_COL] == target_value
            df_nbr = (
                df.loc[item_mask & target_mask]["nbr"]
                .value_counts(normalize=True)
                .sort_index()
                .to_frame()
                .reset_index()
            )
            df_nbr[TARGET_COL] = target_value
            nbrs.append(df_nbr)

            notnull_mask = df["nbr"].notnull()
            total = df.loc[item_mask & target_mask & notnull_mask].shape[0]
            totals.append(total)

        df_nbr = pd.concat(nbrs)

        # Display the distribution as a bar chart.
        sns.barplot(
            df_nbr.query("nbr > 1"),
            x="nbr",
            y="proportion",
            hue=TARGET_COL,
            ax=ax,
        )

        # Add total to the legend
        if not df_nbr.query("nbr > 1").empty:
            l = ax.legend()
            for idx, label in enumerate(l.get_texts()):
                label.set_text(f"{label.get_text()} (total: {totals[idx]:,})")
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        # Add post-hoc labels to the bar
        for idx, container in enumerate(ax.containers):
            target_mask = df[TARGET_COL] == idx
            labels = (
                df.loc[item_mask & target_mask]["nbr"]
                .value_counts(normalize=False)
                .sort_index()
                .values
            )[1:]  # remove nbr = "1"
            ax.bar_label(container, labels)

        ax.yaxis.label.set_visible(False)
        ax.xaxis.label.set_visible(False)
        ax.set_title(item)

    plt.tight_layout()


def _melt_dataframe(df, group_col1, group_col2):
    item_cols = get_group_cols(group_col1)
    nbr_cols = get_group_cols(group_col2)
    dfs = []
    for target_value in [0, 1]:
        df_ = df.loc[df[TARGET_COL] == target_value]
        df_item = df_.melt(value_vars=item_cols, value_name=group_col1)
        df_nbr = df_.melt(value_vars=nbr_cols, value_name=group_col2)
        df_ = df_item[[group_col1]].join(df_nbr[[group_col2]])
        df_[TARGET_COL] = target_value
        dfs.append(df_)
    return pd.concat(dfs)


def plot_price_distribution(df, title=None, figsize=None):
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=figsize)
    palette = ["blue", "orange"]

    for idx, ax in enumerate(axes):
        df_ = df.loc[df[TARGET_COL] == idx]
        mean = df_["total_price_"].mean()
        std = df_["total_price_"].std()
        sns.kdeplot(
            df_["total_price_"],
            fill=True,
            alpha=0.9,
            ax=ax,
            color=palette[idx],
        )
        y_bottom, y_top = ax.get_ylim()
        ax.vlines(
            mean,
            y_bottom,
            y_top,
            color="r",
            linestyle="-",
            linewidth=1,
            label=f"Mean: {mean:.2f}",
        )
        ax.fill_betweenx(
            y=[y_bottom, y_top],
            x1=mean - std,
            x2=mean + std,
            color="gray",
            alpha=0.3,
            label=f"Std Dev: {std:.2f}",
        )
        ax.set_title(f"{TARGET_COL}: {idx}")
        ax.set_xlim([0, df_["total_price_"].quantile(0.99)])
        ax.legend()

    if title:
        fig.suptitle(title)

    plt.tight_layout()


def plot_multiple_price_dist(df, n_most_freq=10):
    freq_items = _get_most_frequent_fraud_items(df, n_most_freq)
    df = _melt_dataframe(df, group_col1="item", group_col2="cash_price")
    for item in freq_items:
        item_mask = df["item"] == item
        df_item = df.loc[item_mask].reset_index()
        df_item["total_price_"] = df_item["cash_price"]
        plot_price_distribution(df_item, title=item, figsize=(4, 4))


def plot_graph(
    df,
    column,
    top=30,
    node_size_coeff=100_000,
    edge_width_coeff=1000,
    node_color="#210070",
    figsize=(20, 20),
    exclude=None,
):
    node_counter, edge_counter = count_nodes_edges(df, column, exclude=exclude)

    top_edges = [tuple_ for tuple_, _ in edge_counter.most_common(n=top)]
    left, right = zip(*top_edges)
    top_items = set([*left, *right])

    node_total = node_counter.total()
    edge_total = edge_counter.total()

    g = nx.Graph()
    g.add_nodes_from(
        [
            (item, {"size": node_size_coeff * total / node_total})
            for item, total in node_counter.items()
            if item in top_items
        ]
    )
    g.add_edges_from(
        [
            (left, right, {"width": edge_width_coeff * total / edge_total})
            for (left, right), total in edge_counter.items()
            if (left, right) in top_edges
        ]
    )

    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.kamada_kawai_layout(g)

    node_size = [g.nodes[node]["size"] for node in list(g)]
    width = [g.edges[edge]["width"] for edge in list(g.edges)]

    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_color,
        alpha=0.9,
        node_size=node_size,
    )
    nx.draw_networkx_edges(
        g,
        pos,
        alpha=0.3,
        edge_color="m",
        width=width,
        node_size=node_size,
    )
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(g, pos, font_size=9, bbox=label_options)
    plt.tight_layout()


def count_nodes_edges(df, column, exclude=None):
    node_counter, edge_counter = Counter(), Counter()

    cols = get_group_cols(column)
    for row in df[cols].values:
        nodes = [node for node in row if node is not pd.NA and node != exclude]
        node_counter.update(nodes)

        edge_counter.update(
            [tuple(sorted(edge)) for edge in combinations(set(nodes), 2)]
        )

    return node_counter, edge_counter


def get_fraud_ratio(df, column):
    counter_valid, _ = count_nodes_edges(
        df.loc[df[TARGET_COL] == 0],
        column=column,
    )
    counter_fraud, _ = count_nodes_edges(
        df.loc[df[TARGET_COL] == 1],
        column=column,
    )

    total_valid = pd.Series(counter_valid, name="total_valid").to_frame()
    total_fraud = pd.Series(counter_fraud, name="total_fraud").to_frame()
    total = total_valid.join(total_fraud).convert_dtypes()
    total["fraud_ratio"] = (
        total["total_fraud"] / (total["total_valid"] + total["total_fraud"])
    ).fillna(0)

    mask = total["fraud_ratio"] > 0
    total = total.loc[mask].sort_values("fraud_ratio", ascending=False)

    return total


def plot_fraud_ratio(df, column, figsize=None):
    palette = ["blue", "coral"]
    df_total = get_fraud_ratio(df, column)

    figsize = figsize or (12, 15)
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twiny()
    sns.lineplot(
        df_total,
        y=df_total.index,
        x="fraud_ratio",
        orient="y",
        color=palette[1],
        marker="o",
        ax=ax2,
    )
    sns.barplot(
        df_total,
        y=df_total.index,
        x="total_fraud",
        orient="y",
        color=palette[0],
        ax=ax,
    )
    avg_fraud = df[TARGET_COL].mean()  # we take the global df for this average
    ax2.vlines(
        avg_fraud,
        ymin=0,
        ymax=df_total.shape[0],
        linestyle="--",
        color="red",
        label=f"Average fraud ratio ({avg_fraud:.4f})",
    )
    ax2.legend()

    ax2.spines["bottom"].set_color(palette[0])
    ax.tick_params(axis="x", colors=palette[0], labelsize=15)
    ax.xaxis.label.set_color(palette[0])
    ax.xaxis.label.set_size(15)
    ax.bar_label(ax.containers[0])

    ax2.spines["top"].set_color(palette[1])
    ax2.tick_params(axis="x", colors=palette[1], labelsize=15)
    ax2.xaxis.label.set_color(palette[1])
    ax2.xaxis.label.set_size(15)

    ax2.grid()
    plt.tight_layout()
