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
