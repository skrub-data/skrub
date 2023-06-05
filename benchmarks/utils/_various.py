from pathlib import Path
from typing import List

import pandas as pd


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
