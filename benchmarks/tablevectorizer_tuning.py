"""
Performs a GridSearch to find the best parameters for the TableVectorizer
among a selection.
"""

import logging
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from dirty_cat import TableVectorizer
from dirty_cat.datasets import (
    fetch_open_payments,
    fetch_drug_directory,
    fetch_road_safety,
    fetch_midwest_survey,
    fetch_medical_charge,
    fetch_employee_salaries,
    fetch_traffic_violations,
)

from pathlib import Path
from functools import wraps
from datetime import datetime
from typing import List, Tuple


def get_classification_datasets() -> List[Tuple[dict, str]]:
    return [
        (fetch_open_payments(), "open_payments"),
        # (fetch_drug_directory(), 'drug_directory),
        (fetch_road_safety(), "road_safety"),
        (fetch_midwest_survey(), "midwest_survey"),
        (fetch_traffic_violations(), "traffic_violations"),
    ]


def get_regression_datasets() -> List[Tuple[dict, str]]:
    return [
        (fetch_medical_charge(), "medical_charge"),
        (fetch_employee_salaries(), "employee_salaries"),
    ]


def get_dataset(info) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(info["path"], **info["read_csv_kwargs"])
    y = df[info["y"]]
    X = df.drop(info["y"], axis=1).astype(str)
    return X, y


def set_logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging_level = logging.DEBUG

        logger = logging.getLogger()
        logger.setLevel(logging_level)

        formatter = logging.Formatter("%(asctime)s - [%(levelname)s] %(message)s")
        formatter.datefmt = "%m/%d/%Y %H:%M:%S"

        path = Path(__file__).parent / f"tuning_{str(datetime.now())[:10]}.log"

        fh = logging.FileHandler(filename=path, mode="w")
        fh.setLevel(logging_level)
        fh.setFormatter(formatter)

        # sh = logging.StreamHandler(sys.stdout)
        # sh.setLevel(logging_level)
        # sh.setFormatter(formatter)

        logger.addHandler(fh)
        # logger.addHandler(sh)

        return func(*args, **kwargs)

    return wrapper


@set_logging
def main():
    logging.info("Launching !")

    card_possibilities = [20, 30, 40, 50]
    n_comp_possibilities = [10, 30, 50]

    logging.debug("Creating pipelines")
    regression_pipeline = Pipeline(
        [
            ("tv", TableVectorizer()),
            ("estimator", RandomForestRegressor()),
        ]
    )
    classification_pipeline = Pipeline(
        [
            ("tv", TableVectorizer()),
            ("estimator", RandomForestClassifier()),
        ]
    )

    logging.debug(
        f"With cardinality possibilities: {card_possibilities} "
        f"and n_components possibilities: {n_comp_possibilities}"
    )
    for pipeline, datasets in zip(
        [
            regression_pipeline,
            classification_pipeline,
        ],
        [
            get_regression_datasets(),
            get_classification_datasets(),
        ],
    ):
        for info, name in datasets:
            X, y = get_dataset(info)
            if name != "traffic_violations":
                continue

            csv_path = Path(".").resolve() / f"{name}_results.csv"
            if csv_path.exists():
                # If the results already exist, we'll skip to the next
                logging.debug(f"Skipping {name} as {csv_path!s} was found")
                continue

            logging.debug(f"Running search on {name}")
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid={
                    "sv__cardinality_threshold": card_possibilities,
                    "sv__high_card_str_transformer__n_components": n_comp_possibilities,
                },
                n_jobs=30,
            )
            grid.fit(X, y)

            df = pd.DataFrame(grid.cv_results_)
            df.to_csv(csv_path)
            logging.info(f"Saved search results in {csv_path!s}")


if __name__ == "__main__":
    main()
