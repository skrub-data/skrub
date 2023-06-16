from pathlib import Path

from skrub.datasets import (
    fetch_open_payments,
    fetch_drug_directory,
    fetch_road_safety,
    fetch_midwest_survey,
    fetch_medical_charge,
    fetch_employee_salaries,
    fetch_traffic_violations,
)

from skrub.datasets._fetching import DatasetAll

import pandas as pd


def find_result(bench_name: str) -> Path:
    return choose_file(find_results(bench_name))


def find_results(bench_name: str) -> list[Path]:
    """
    Returns the list of results in the results' directory.
    """
    results_dir = Path(__file__).parent.parent / "results"
    return [
        file
        for file in results_dir.iterdir()
        if file.stem.startswith(bench_name) and file.suffix == ".parquet"
    ]


def choose_file(results: list[Path]) -> Path:
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
            df = pd.read_parquet(file)
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


def get_classification_datasets() -> list[tuple[DatasetAll, str]]:
    return [
        (fetch_open_payments(), "open_payments"),
        (fetch_drug_directory(), "drug_directory"),
        (fetch_road_safety(), "road_safety"),
        (fetch_midwest_survey(), "midwest_survey"),
        (fetch_traffic_violations(), "traffic_violations"),
    ]


def get_regression_datasets() -> list[tuple[DatasetAll, str]]:
    return [
        (fetch_medical_charge(), "medical_charge"),
        (fetch_employee_salaries(), "employee_salaries"),
    ]


def get_dataset(info: tuple[DatasetAll, str]) -> tuple[pd.DataFrame, pd.Series]:
    y = info.y
    X = info.X
    return X, y
