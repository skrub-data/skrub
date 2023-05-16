from pathlib import Path
from typing import List

import pandas as pd


def fetch_data(dataset_name):
    """Fetch datasets from https://github.com/chu-data-lab/AutomaticFuzzyJoin/tree/master/src/autofj/benchmark
    """
    repository = "chu-data-lab/AutomaticFuzzyJoin"
    base_url = f"https://raw.githubusercontent.com/{repository}/master/src/autofj/benchmark/{dataset_name}"  # noqa
    left = pd.read_csv(f"{base_url}/left.csv")
    right = pd.read_csv(f"{base_url}/right.csv")
    gt = pd.read_csv(f"{base_url}/gt.csv")
    return left, right, gt


def evaluate(pred_joins, gt_joins):
    """Evaluate the performance of fuzzy joins

    Parameters
    ----------
    pred_joins: list
        A list of tuple pairs (id_l, id_r) that are predicted to be matches

    gt_joins:
        The ground truth matches

    Returns
    -------
    precision: float
        Precision score

    recall: float
        Recall score

    f1: float
        F1 score
    """
    pred = {(le, ri) for le, ri in pred_joins}
    gt = {(le, ri) for le, ri in gt_joins}

    tp = pred.intersection(gt)
    precision = len(tp) / len(pred)
    recall = len(tp) / len(gt)
    # print('Precision', precision, 'Recall', recall)
    if precision > 0 or recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


def find_result(bench_name: str) -> Path:
    return choose_file(find_results(bench_name))


def find_results(bench_name: str) -> List[Path]:
    """
    Returns the list of results in the results' directory.
    """
    results_dir = Path(__file__).parent.parent / "results"
    return [
        file
        for file in results_dir.iterdir()
        if file.stem.startswith(bench_name) and file.suffix == ".csv"
    ]


def choose_file(results: List[Path]) -> Path:
    """
    Given a list of files, chooses one based on these rules:
    - If there are no files to choose from, exit the program
    - If there's only one, return this one
    - If there are multiple, prompt the user to choose one
    """
    if len(results) == 0:
        print("No results file to choose from, exiting...")
        exit()
    elif len(results) == 1:
        return results[0]
    else:
        for i, file in enumerate(results):
            # Read the result file to get its dimensions
            df = pd.read_csv(file)
            if "iter" not in df.columns:
                print(f"Invalid file {file.name!r}, skipping.")
                continue
            n_iter_per_xp = df["iter"].max() + 1
            repeat = df.shape[0] // n_iter_per_xp

            bench_name, date = file.stem.split("-")
            print(
                f"{i + 1}) "
                f"{date[:4]}-{date[4:6]}-{date[6:]} - "
                f"{df.shape[0]}x{repeat} experiments "
            )
        choice = input("Choose the result to display: ")
        if not choice.isnumeric() or (int(choice) - 1) not in range(len(results)):
            print(f"Invalid choice {choice!r}, exiting.")
            exit()
        return results[int(choice) - 1]
