from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

import skrub
import skrub.datasets
from skrub import selectors as s


def create_expression_report():
    dataset = skrub.datasets.fetch_credit_fraud()

    products = skrub.var("products", dataset.products)
    baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_x()
    fraud_flags = skrub.var(
        "fraud_flags", dataset.baskets["fraud_flag"]
    ).skb.mark_as_y()

    products = products[products["basket_ID"].isin(baskets["ID"])]

    product_vectorizer = skrub.TableVectorizer(
        high_cardinality=skrub.StringEncoder(n_components=5)
    )

    vectorized_products = products.skb.apply(
        product_vectorizer, cols=s.all() - "basket_ID"
    )

    aggregated_products = (
        vectorized_products.groupby("basket_ID").agg("mean").reset_index()
    )
    baskets = baskets.merge(
        aggregated_products, left_on="ID", right_on="basket_ID"
    ).drop(columns=["ID", "basket_ID"])

    predictions = baskets.skb.apply(HistGradientBoostingClassifier(), y=fraud_flags)

    predictions.skb.full_report(
        output_dir=Path(__file__).parent
        / "_build"
        / "html"
        / "_static"
        / "credit_fraud_report",
        overwrite=True,
    )
