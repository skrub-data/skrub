from pathlib import Path

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

import skrub
import skrub.datasets
from skrub import TableVectorizer
from skrub import selectors as s


def create_credit_fraud_report():
    output_dir = (
        Path(__file__).parent / "_build" / "html" / "_static" / "credit_fraud_report"
    )
    if (output_dir / "index.html").exists():
        return output_dir

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

    return output_dir


def create_employee_salaries_report():
    output_dir = (
        Path(__file__).parent
        / "_build"
        / "html"
        / "_static"
        / "employee_salaries_report"
    )
    if (output_dir / "index.html").exists():
        return output_dir

    dataset = skrub.datasets.fetch_employee_salaries(split="train").employee_salaries
    data_var = skrub.var("data", dataset)
    X = data_var.drop("current_annual_salary", axis=1).skb.mark_as_X()
    y = data_var["current_annual_salary"].skb.mark_as_y()

    vectorizer = TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    hgb = HistGradientBoostingRegressor()
    predictor = X_vec.skb.apply(hgb, y=y)

    predictor.skb.full_report(
        output_dir=output_dir,
        overwrite=True,
        open=False,
    )

    return output_dir


def create_data_ops_report():
    credit_fraud_folder = create_credit_fraud_report()
    employee_salary_folder = create_employee_salaries_report()

    assert (credit_fraud_folder / "index.html").exists()
    assert (employee_salary_folder / "index.html").exists()
