from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

import skrub
import skrub.datasets
from skrub import selectors as s


def create_expression_report():
    output_dir = (
        Path(__file__).parent / "_build" / "html" / "_static" / "credit_fraud_report"
    )
    if output_dir.exists():
        return

    dataset = skrub.datasets.fetch_credit_fraud()

    products = skrub.var("products", dataset.products)
    baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_X()
    fraud_flags = skrub.var(
        "fraud_flags", dataset.baskets["fraud_flag"]
    ).skb.mark_as_y()

    products = products[products["basket_ID"].isin(baskets["ID"])]
    products = products.assign(
        total_price=products["Nbr_of_prod_purchas"] * products["cash_price"]
    )
    n = skrub.choose_int(5, 15, log=True, name="n_components")
    encoder = skrub.choose_from(
        {
            "MinHash": skrub.MinHashEncoder(n_components=n),
            "LSA": skrub.StringEncoder(n_components=n),
        },
        name="encoder",
    )
    vectorizer = skrub.TableVectorizer(high_cardinality=encoder)

    vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")

    aggregated_products = (
        vectorized_products.groupby("basket_ID").agg("mean").reset_index()
    )
    baskets = baskets.merge(
        aggregated_products, left_on="ID", right_on="basket_ID"
    ).drop(columns=["ID", "basket_ID"])

    hgb = HistGradientBoostingClassifier(
        learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
    )
    predictions = baskets.skb.apply(hgb, y=fraud_flags)

    predictions.skb.full_report(
        output_dir=output_dir,
        overwrite=True,
        open=False,
    )
