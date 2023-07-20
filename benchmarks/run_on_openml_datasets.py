"""
This script runs the canonical skrub pipeline on all OpenML datasets.
It can be used to check that we can deal with any dataset without failing.
It can also be used to compare our scores to OpenML scores uploaded by other users,
using the --compare_scores flag (this is slow).
"""


import openml
import os
import numpy as np
from skrub import TableVectorizer, MinHashEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import cross_val_score
import argparse

# argparse
parser = argparse.ArgumentParser()
# add max number of rows
parser.add_argument("--max_rows", type=int, default=500000)
parser.add_argument("--min_rows", type=int, default=10)
parser.add_argument("--max_features", type=int, default=1000)
# int or None
parser.add_argument("--rows", type=int, default=500)
# add an option to compare scores to OpenML
parser.add_argument("--compare_scores", action="store_true")
# n_jobs
parser.add_argument("--n_jobs", type=int, default=1)
# openml cache directory
parser.add_argument("--cache_directory", type=str, default="~/.openml/cache")
args = parser.parse_args()

# openml.config.cache_directory = os.path.expanduser(args.cache_directory)

types = [1, 2]  # classification and regression

pipe_clf = Pipeline(
    [
        ("vectorizer", TableVectorizer(high_card_cat_transformer=MinHashEncoder())),
        ("classifier", HistGradientBoostingClassifier()),
    ]
)

pipe_reg = Pipeline(
    [
        ("vectorizer", TableVectorizer(high_card_cat_transformer=MinHashEncoder())),
        ("regressor", HistGradientBoostingRegressor()),
    ]
)

errors = {}
low_scores = {}

for type in types:
    if type == 1:
        pipe = pipe_clf
    else:
        pipe = pipe_reg
    for task_id in openml.tasks.list_tasks(type=type):
        try:
            task = openml.tasks.get_task(task_id, download_data=False)
            dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
            # check if it's not too big
            if dataset.qualities["NumberOfInstances"] > 500000:
                print(f"Skipping task {task_id} because it has too many instances")
                continue
            if dataset.qualities["NumberOfInstances"] < 10:
                print(f"Skipping task {task_id} because it has too few instances")
                continue
            if dataset.qualities["NumberOfFeatures"] > 1000:
                print(f"Skipping task {task_id} because it has too many features")
                continue
        except Exception as e:
            print(f"Skipping task {task_id} because it could not be fetched")
            print(e)
            continue
        print(f"Running task {task_id}")
        try:
            # get results and compare to OpenML
            if type == 1:
                metric = "predictive_accuracy"
            elif type == 2:
                metric = "mean_absolute_error"
            if not args.compare_scores:
                try:
                    X, y, categorical_indicator, attribute_names = dataset.get_data(
                        dataset_format="dataframe", target=task.target_name
                    )
                except Exception as e:
                    print(f"Skipping task {task_id} because it could not be downloaded")
                    print(e)
                    continue
                # fit on a subset of the data to check that it works
                n = args.rows
                if X.shape[0] < n:
                    n = X.shape[0]
                X = X.sample(n=n, random_state=42)
                y = y[X.index]
                scores = cross_val_score(pipe, X, y, cv=2)
                print(
                    f"Task {task_id} scored {np.mean(scores)} using"
                    f" {n}/{dataset.qualities['NumberOfInstances']} rows"
                )
            else:
                run = openml.runs.run_model_on_task(
                    pipe, task, avoid_duplicate_runs=False
                )
                scores = list(run.fold_evaluations[metric][0].values())
                # scores is an OrderedDict of OrderedDicts
                print(f"Task {task_id} scored {np.mean(scores)} +- {np.std(scores)}")
            evals = openml.evaluations.list_evaluations(
                function=metric, tasks=[task_id], output_format="dataframe"
            )

            print(f"OpenML scores on {len(evals)} runs (on the full dataset):")
            print("25% percentile:", np.percentile(evals.value, 25))
            print("50% percentile:", np.percentile(evals.value, 50))
            print("75% percentile:", np.percentile(evals.value, 75))
            # warn if our score is below the 25% percentile
            if args.compare_scores:
                if type == 1 and np.mean(scores) < np.percentile(evals.value, 25):
                    print("-----------------------------------")
                    print("WARNING: our score is below the 25% percentile")
                    print("-----------------------------------")
                    low_scores[task_id] = np.mean(scores)
                if type == 2 and np.mean(scores) > np.percentile(evals.value, 25):
                    print("-----------------------------------")
                    print("WARNING: our score is above the 25% percentile")
                    print("-----------------------------------")
                    low_scores[task_id] = np.mean(scores)
        except Exception as e:
            print(f"Skipping task {task_id} because of error {e}")
            errors[task_id] = e
            continue

print(f"Finished! {len(errors)} errors occurred")
print("Task IDs with errors:")
print(errors.keys())
if args.compare_scores:
    print(
        f"{len(low_scores)} tasks have scores below the 25% percentile of OpenML runs"
    )
    print("Task IDs with low scores:")
    print(low_scores.keys())
