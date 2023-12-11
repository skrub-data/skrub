"""
This script runs the canonical skrub pipeline on all OpenML datasets.
It can be used to check that we can deal with any dataset without failing.
It can also be used to compare our scores to OpenML scores uploaded by other users,
using the `--compare_scores` flag (this is slow).
"""
from collections import Counter
import openml
import os
import numpy as np
from benchmarks.utils import default_parser
from skrub import TableVectorizer, MinHashEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import cross_val_score
import argparse
from loguru import logger

# argparse
parser = argparse.ArgumentParser(parents=[default_parser])
# add max number of rows
parser.add_argument("--max_rows", type=int, default=500000)
parser.add_argument("--min_rows", type=int, default=10)
parser.add_argument("--max_features", type=int, default=1000)
parser.add_argument(
    "--row_limit",
    type=int,
    default=500,
    help="Random sample size taken from the dataset to assert the pipeline works.",
)
# add an option to compare scores to OpenML
parser.add_argument(
    "--compare_scores",
    action="store_true",
    help="Compare our scores to the ones available on OpenML (this is slow)",
)
parser.add_argument("--n_jobs", type=int, default=1)
parser.add_argument("--cache_directory", type=str, default="~/.openml/cache")
args = parser.parse_args()

openml.config.cache_directory = os.path.expanduser(args.cache_directory)

classification_pipeline = Pipeline(
    [
        ("vectorizer", TableVectorizer(high_cardinality_transformer=MinHashEncoder())),
        ("classifier", HistGradientBoostingClassifier()),
    ]
)

regression_pipeline = Pipeline(
    [
        ("vectorizer", TableVectorizer(high_cardinality_transformer=MinHashEncoder())),
        ("regressor", HistGradientBoostingRegressor()),
    ]
)

errors = {}
low_scores = {}
constraint_error_template = "Skipping task {id} because {reason}"

for type_id, problem, pipeline, metric in [
    (1, "classification", classification_pipeline, "predictive_accuracy"),
    (2, "regression", regression_pipeline, "mean_absolute_error"),
]:
    for task_id in openml.tasks.list_tasks(type=type_id):
        try:
            task = openml.tasks.get_task(task_id, download_splits=True)
            dataset = openml.datasets.get_dataset(
                task.dataset_id,
                download_data=False,
                download_qualities=True,
                download_features_meta_data=False,
            )
            # check if it's not too big
            if dataset.qualities["NumberOfInstances"] > args.max_rows:
                logger.debug(
                    constraint_error_template.format(
                        id=task_id, reason="it has too many instances"
                    )
                )
                continue
            if dataset.qualities["NumberOfInstances"] < args.min_rows:
                logger.debug(
                    constraint_error_template.format(
                        id=task_id, reason="it has too few instances"
                    )
                )
                continue
            if dataset.qualities["NumberOfFeatures"] > args.max_features:
                logger.debug(
                    constraint_error_template.format(
                        id=task_id, reason="it has too many features"
                    )
                )
                continue
            if dataset.qualities["NumberOfFeatures"] < 1:
                logger.debug(
                    constraint_error_template.format(
                        id=task_id, reason="it has too few features"
                    )
                )
                continue
        except Exception as e:
            logger.warning(
                constraint_error_template.format(
                    id=task_id, reason=f"it could not be fetched, exception: {e}"
                )
            )
            continue
        logger.info(f"Running task {task_id}")
        try:
            # get results and compare to OpenML
            if args.compare_scores:
                run = openml.runs.run_model_on_task(
                    pipeline, task, avoid_duplicate_runs=False
                )
                scores = list(run.fold_evaluations[metric][0].values())
                mean = np.mean(scores)
                std = np.std(scores)
                # scores is an OrderedDict of OrderedDicts
                logger.info(f"Task {task_id} scored {mean} +- {std}. ")
            else:
                try:
                    X, y, categorical_indicator, attribute_names = dataset.get_data(
                        dataset_format="dataframe", target=task.target_name
                    )
                except Exception as e:
                    logger.warning(
                        constraint_error_template.format(
                            id=task_id,
                            reason=f"it could not be fetched, exception: {e}",
                        )
                    )
                    continue
                # fit on a subset of the data to check that it works
                n = args.row_limit
                if X.shape[0] < n:
                    n = X.shape[0]
                X = X.sample(n=n, random_state=0)
                y = y[X.index]
                scores = cross_val_score(pipeline, X, y, cv=2)
                mean = np.mean(scores)
                std = np.std(scores)
                logger.info(
                    f"Task {task_id} scored {mean} +- {std} using "
                    f"{n}/{dataset.qualities['NumberOfInstances']} rows. "
                )
            evals = openml.evaluations.list_evaluations(
                function=metric, tasks=[task_id], output_format="dataframe"
            )
            if len(evals) > 0:
                percentiles = {p: np.percentile(evals.value, p) for p in {25, 50, 75}}
                logger.info(
                    f"OpenML scores on {len(evals)} runs (on the full dataset): "
                    + " ; ".join(
                        f"{p}% percentile: {value}" for p, value in percentiles.items()
                    )
                )
                if args.compare_scores:
                    if (problem == "classification" and mean < percentiles[25]) or (
                        problem == "regression" and mean > percentiles[25]
                    ):
                        logger.warning(
                            f"Our score is below the 25% percentile on {task_id}"
                        )
                        low_scores[task_id] = np.mean(scores)
            else:
                logger.debug(f"No OpenML scores available for {task_id}")
        except Exception as e:
            logger.warning(
                constraint_error_template.format(
                    id=task_id, reason=f"of an unhandled error: {e}."
                )
            )
            errors[task_id] = str(e)
            continue

logger.info(f"Finished! ")
logger.error(f"{len(errors)} tasks with errors: {set(errors.keys())}")
# print all unique errors
errors_counter = Counter(errors.values())
for error, count in errors_counter.items():
    logger.error(f"Error: {error}, {count} times")

if args.compare_scores:
    logger.info(
        f"{len(low_scores)} tasks with low scores "
        f"(25% percentile on OpenML runs: {set(low_scores.keys())}"
    )
