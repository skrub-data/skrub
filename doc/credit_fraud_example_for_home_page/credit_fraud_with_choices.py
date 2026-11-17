from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

import skrub


def create_report(output_dir):
    """Build the credit fraud pipeline and write its report and graph.

    The report and the graph come from the same pipeline so that the graph, the
    interactive report and the code snippet on the homepage always describe the
    same pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset = skrub.datasets.fetch_credit_fraud(split="train")

    data_source = skrub.var(
        "data_source", {"orders": dataset.baskets, "products": dataset.products}
    )

    def load_orders(data_source, order_ids=None):
        if order_ids is None:
            return data_source["orders"]
        return (
            data_source["orders"].set_index("ID").loc[order_ids].reset_index(drop=False)
        )

    query = skrub.var("order_IDs", dataset["baskets"]["ID"])
    orders = data_source.skb.apply_func(load_orders, query)
    order_ids = orders[["ID"]].skb.mark_as_X()
    fraud_flags = orders["fraud_flag"].skb.mark_as_y()

    def load_products(order_ids, data_source):
        products = data_source["products"]
        return products[products["basket_ID"].isin(order_ids["ID"])]

    encoder = skrub.choose_from(
        {"bow": skrub.StringEncoder(), "llm": skrub.LLMEncoder()}
    )
    vectorized_products = order_ids.skb.apply_func(
        load_products, data_source
    ).skb.apply(
        skrub.TableVectorizer(high_cardinality=encoder),
        exclude_cols="basket_ID",
    )

    def join_orders_products(orders, vectorized_products, agg_func):
        return orders.merge(
            vectorized_products.groupby("basket_ID").agg(agg_func).reset_index(),
            left_on="ID",
            right_on="basket_ID",
        ).drop(columns=["ID", "basket_ID"])

    features = order_ids.skb.apply_func(
        join_orders_products,
        vectorized_products,
        agg_func=skrub.choose_from(["mean", "sum"]),
    )
    pred = features.skb.apply(
        HistGradientBoostingClassifier(
            learning_rate=skrub.choose_float(0.01, 0.5, log=True)
        ),
        y=fraud_flags,
    )

    pred.skb.report(
        output_dir=output_dir,
        overwrite=True,
        open=False,
        title="Credit fraud prediction",
    )

    graph = pred.skb.draw_graph()
    (output_dir / "credit_fraud_graph.svg").write_bytes(graph.svg)
    (output_dir / "credit_fraud_graph.png").write_bytes(graph.png)
