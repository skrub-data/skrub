import pandas as pd
from pathlib import Path

from skrub.datasets._utils import get_data_dir


def get_local_data(
    dataset_name: str,
    data_home: Path | str | None = None,
    data_directory: str | None = None,
):
    """Get the path to the local datasets."""
    data_directory = get_data_dir(data_directory, data_home)
    left_path = str(data_directory) + f"/left_{dataset_name}.parquet"
    right_path = str(data_directory) + f"/right_{dataset_name}.parquet"
    gt_path = str(data_directory) + f"/gt_{dataset_name}.parquet"
    data_directory.mkdir(parents=True, exist_ok=True)
    file_paths = [
        file
        for file in data_directory.iterdir()
        if file.name.endswith(f"{dataset_name}.parquet")
    ]
    print(file_paths)
    return left_path, right_path, gt_path, file_paths


def fetch_data(
    dataset_name: str,
    save: bool = True,
    data_home: Path | str | None = None,
    data_directory: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch datasets from https://github.com/Yeye-He/Auto-Join/tree/master/autojoin-Benchmark  # noqa

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to download.

    save: bool, default=true
        Wheter to save the datasets locally.

    data_home: Path or str, optional
        The path to the root data directory.
        By default, will point to the skrub data directory.

    data_directory: str, optional
        The name of the subdirectory in which data is stored.

    Returns
    -------
    left: pd.DataFrame
        Left dataset.

    right: pd.DataFrame
        Right dataset.

    gt: pd.DataFrame
        Ground truth dataset.
    """
    left_path, right_path, gt_path, file_paths = get_local_data(
        dataset_name, data_home, data_directory
    )
    if len(file_paths) == 0:
        repository = "Yeye-He/Auto-Join"
        dataset_name = dataset_name.replace(" ", "%20")
        base_url = base_url = (
            "https://raw.githubusercontent.com/"
            f"{repository}/master/autojoin-Benchmark/{dataset_name}"
        )
        left = pd.read_csv(f"{base_url}/source.csv")
        right = pd.read_csv(f"{base_url}/target.csv")
        gt = pd.read_csv(f"{base_url}/ground%20truth.csv")
        if save is True:
            left.to_parquet(left_path)
            right.to_parquet(right_path)
            gt.to_parquet(gt_path)
    else:
        left = pd.read_parquet(left_path)
        right = pd.read_parquet(right_path)
        gt = pd.read_parquet(gt_path)
    return left, right, gt


def fetch_big_data(
    dataset_name: str,
    data_type: str = "Dirty",
    save: bool = True,
    data_home: Path | str | None = None,
    data_directory: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch datasets from https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  # noqa

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to download.

    data_type: str
        The type of data to be downloaded.
        Options are {'Dirty', 'Structured', 'Textual'}.

    save: bool, default=true
        Wheter to save the datasets locally.

    data_home: Path or str, optional
        The path to the root data directory.
        By default, will point to the skrub data directory.

    data_directory: str, optional
        The name of the subdirectory in which data is stored.

    Returns
    -------
    left: pd.DataFrame
        Left dataset.

    right: pd.DataFrame
        Right dataset.

    gt: pd.DataFrame
        Ground truth dataset.
    """
    link = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/"
    left_path, right_path, gt_path, file_paths = get_local_data(
        dataset_name, data_home, data_directory
    )
    if len(file_paths) == 0:
        test_idx = pd.read_csv(f"{link}/{data_type}/{dataset_name}/exp_data/test.csv")
        train_idx = pd.read_csv(f"{link}/{data_type}/{dataset_name}/exp_data/train.csv")
        valid_idx = pd.read_csv(f"{link}/{data_type}/{dataset_name}/exp_data/valid.csv")
        idx = pd.concat([test_idx, train_idx, valid_idx], ignore_index=True)
        idx = idx[idx["label"] == 1].reset_index()

        left = pd.read_csv(
            f"{link}/{data_type}/{dataset_name}/exp_data/tableA.csv"
        ).iloc[idx["ltable_id"], 1]
        left = left.rename("title")
        right = pd.read_csv(
            f"{link}/{data_type}/{dataset_name}/exp_data/tableB.csv"
        ).iloc[idx["rtable_id"], 1]
        right = right.rename("title")
        left = left.reset_index(drop=True).reset_index()
        right = right.reset_index(drop=True).reset_index()
        gt = pd.merge(left, right, on="index", suffixes=("_l", "_r"))
        if save is True:
            left.to_parquet(left_path)
            right.to_parquet(right_path)
            gt.to_parquet(gt_path)
    else:
        left = pd.read_parquet(left_path)
        right = pd.read_parquet(right_path)
        gt = pd.read_parquet(gt_path)
    return left, right, gt


def evaluate(pred_joins, gt_joins):
    """Evaluate the performance of fuzzy joins

    Parameters
    ----------
    pred_joins: list
        A list of tuple pairs (id_l, id_r) that are predicted to be matches

    gt_joins: list
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
    if precision > 0 or recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1
