import pandas as pd
import os
from pathlib import Path


def get_local_data(dataset_name):
    """ Get the path to the local datasets. """
    module_path = Path(os.path.dirname(__file__)).resolve()
    data_dir = module_path / "data"
    left_path = str(data_dir) + f"/left_{dataset_name}.parquet"
    right_path = str(data_dir) + f"/right_{dataset_name}.parquet"
    gt_path = str(data_dir) + f"/gt_{dataset_name}.parquet"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_paths = [
        file
        for file in data_dir.iterdir()
        if file.name.endswith(f"{dataset_name}.parquet")
    ]
    return left_path, right_path, gt_path, file_paths


def fetch_data(dataset_name, save=True):
    """Fetch datasets from https://github.com/Yeye-He/Auto-Join/tree/master/autojoin-Benchmark

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to download.

    save: bool, default=true
        Wheter to save the datasets locally.

    Returns
    -------
    left: pd.DataFrame
        Left dataset.

    right: pd.DataFrame
        Right dataset.

    gt: pd.DataFrame
        Ground truth dataset.
    """
    left_path, right_path, gt_path, file_paths = get_local_data(dataset_name)
    if len(file_paths) == 0:
        repository = "Yeye-He/Auto-Join"
        dataset_name = dataset_name.replace(' ', '%20')
        base_url = base_url = (
            f"https://raw.githubusercontent.com/"
            f"{repository}/master/autojoin-Benchmark/{dataset_name}"
        )
        left = pd.read_csv(f"{base_url}/source.csv")
        right = pd.read_csv(f"{base_url}/target.csv")
        gt = pd.read_csv(f"{base_url}/ground%20truth.csv")
        left.to_parquet(left_path)
        right.to_parquet(right_path)
        gt.to_parquet(gt_path)
    else:
        left = pd.read_parquet(left_path)
        right = pd.read_parquet(right_path)
        gt = pd.read_parquet(gt_path)
    return left, right, gt


def fetch_big_data(dataset_name, data_type='Dirty', save=True):
    """Fetch datasets from https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to download.

    data_type: str
        The type of data to be downloaded.
        Options are {'Dirty', 'Structured', 'Textual'}.

    save: bool, default=true
        Wheter to save the datasets locally.

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
    left_path, right_path, gt_path, file_paths = get_local_data(dataset_name)
    if len(file_paths) == 0:
        idx1 = pd.read_csv(f"{link}/{data_type}/{dataset_name}/exp_data/test.csv")
        idx2 = pd.read_csv(f"{link}/{data_type}/{dataset_name}/exp_data/train.csv")
        idx3 = pd.read_csv(f"{link}/{data_type}/{dataset_name}/exp_data/valid.csv")
        idx = pd.concat([idx1, idx2, idx3], ignore_index=True)
        idx = idx[idx["label"] == 1].reset_index()

        left = pd.read_csv(f"{link}/{data_type}/{dataset_name}/exp_data/tableA.csv").iloc[idx["ltable_id"], 1]
        left = left.rename('title')
        right = pd.read_csv(f"{link}/{data_type}/{dataset_name}/exp_data/tableB.csv").iloc[idx["rtable_id"], 1]
        right = right.rename('title')
        left = left.reset_index(drop=True).reset_index()
        right = right.reset_index(drop=True).reset_index()
        gt = pd.merge(left, right, on='index', suffixes=('_l', '_r'))
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
