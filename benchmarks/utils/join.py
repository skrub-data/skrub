import pandas as pd


def fetch_data(dataset_name):
    """Fetch datasets from https://github.com/Yeye-He/Auto-Join/tree/master/autojoin-Benchmark
    """
    repository = "Yeye-He/Auto-Join"
    if isinstance(dataset_name, str):
        dataset_name = dataset_name.replace(' ', '%20')
        base_url = f"https://raw.githubusercontent.com/{repository}/master/autojoin-Benchmark/{dataset_name}"  # noqa
        left = pd.read_csv(f"{base_url}/source.csv")
        right = pd.read_csv(f"{base_url}/target.csv")
        gt = pd.read_csv(f"{base_url}/ground%20truth.csv")
    elif isinstance(dataset_name, list):
        left = pd.DataFrame()
        right = pd.DataFrame()
        gt = pd.DataFrame()
        for name in dataset_name:
            base_url = f"https://raw.githubusercontent.com/{repository}/master/autojoin-Benchmark/{name}"  # noqa
            left_p = pd.read_csv(f"{base_url}/source.csv")
            right_p = pd.read_csv(f"{base_url}/target.csv")
            gt_p = pd.read_csv(f"{base_url}/ground%20truth.csv")
            left = left.append(left_p)
            right = right.append(right_p)
            gt = gt.append(gt_p)
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
    # print('Precision', precision, 'Recall', recall)
    if precision > 0 or recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1
