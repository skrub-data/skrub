import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from thefuzz.fuzz import partial_ratio, WRatio, ratio
from thefuzz import process
from autofj import AutoFJ
from skrub._fuzzy_join import fuzzy_join


def fetch_data(dataset_name):
    """Fetch datasets from https://github.com/chu-data-lab/AutomaticFuzzyJoin/tree/master/src/autofj/benchmark
    """
    repository = "chu-data-lab/AutomaticFuzzyJoin"
    base_url = f"https://raw.githubusercontent.com/{repository}/master/src/autofj/benchmark/{dataset_name}"  # noqa
    left = pd.read_csv(f"{base_url}/left.csv")
    right = pd.read_csv(f"{base_url}/right.csv")
    gt = pd.read_csv(f"{base_url}/gt.csv")
    return left, right, gt


def thefuzz_merge(df_1, df_2, key1, key2, threshold=90, limit=2, scorer=partial_ratio):
    """
    Merging using thefuzz

    Parameters:
        df_1: the left table to join
        df_2: the right table to join
        key1: key column of the left table
        key2: key column of the right table
        threshold: how close the matches should be to return a match, based on Levenshtein distance
        limit: the amount of matches that will get returned, these are sorted high to low

    Return:
        dataframe with boths keys and matches
    """
    s = df_2[key2].tolist()

    m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit, scorer=scorer))
    df_1["matches"] = m

    m2 = df_1["matches"].apply(
        lambda x: ", ".join([i[0] for i in x if i[1] >= threshold])
    )

    df_1["matches"] = m2

    return df_1[df_1["matches"] != ""]


def test_autofj(left, right, gt, target):
    """Merging using AutomaticFuzzyJoin"""
    autofj = AutoFJ(precision_target=target, verbose=True)
    LR_joins = autofj.join(left, right, id_column="id")

    gt_joins = gt[["id_l", "id_r"]].values
    LR_joins = LR_joins[["id_l", "id_r"]].values
    p, r, f1 = evaluate(LR_joins, gt_joins)
    print("Precision:", p, "Recall:", r, "F1:", f1)
    return LR_joins, gt_joins


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


def fuzzy_join_precision_recall(
    left, right, gt, left_col, right_col, analyzer, ngram_range
):
    joined_fj = fuzzy_join(
        right,
        left,
        left_on=left_col,
        right_on=right_col,
        analyzer=analyzer,
        ngram_range=ngram_range,
        return_score=True,
    )
    pr_list = []
    re_list = []
    f1_list = []
    pr, re, f1 = evaluate(
        list(zip(joined_fj["title_r"], joined_fj["title_l"])),
        list(zip(gt["title_l"], gt["title_r"])),
    )
    pr_list.append(pr)
    re_list.append(re)
    f1_list.append(f1)
    return pr_list, re_list, f1_list


def thefuzz_precision_recall(
    left, right, gt, left_col, right_col, scorer=partial_ratio
):
    pr_list = []
    re_list = []
    f1_list = []
    for threshold in range(60, 99, 15):
        joined = thefuzz_merge(
            left,
            right,
            left_col,
            right_col,
            threshold=threshold,
            limit=1,
            scorer=scorer,
        )
        pr, re, f1 = evaluate(
            list(zip(joined[left_col], joined["matches"])),
            list(zip(gt["title_l"], gt["title_r"])),
        )
        pr_list.append(pr)
        re_list.append(re)
        f1_list.append(f1)
    return pr_list, re_list, f1_list


def autofj_precision_recall(left, right, gt, n_points=20):
    pr_list = []
    re_list = []
    f1_list = []
    for target in np.linspace(0.6, 1, n_points):
        autofj = AutoFJ(precision_target=target, verbose=True)
        LR_joins = autofj.join(left, right, id_column="id")
        if len(LR_joins) != 0:
            gt_joins = gt[["id_l", "id_r"]].values
            LR_joins = LR_joins[["id_l", "id_r"]].values
            p, r, f1 = evaluate(LR_joins, gt_joins)
            pr_list.append(p)
            re_list.append(r)
            f1_list.append(f1)
            # print("Precision:", p, "Recall:", r, "F1:", f1)
    return pr_list, re_list, f1_list


def best_precision_recall(pr_list, re_list, n_bins=13):
    bins = np.digitize(re_list, np.quantile(re_list, np.linspace(0, 1, n_bins)))
    res_pr = np.zeros(n_bins - 1)
    res_re = np.zeros(n_bins - 1)
    for i in range(1, n_bins):
        mask = bins == i
        # print("Recall:", re_list[mask], "Precision:", pr_list[mask])
        if len(pr_list[mask]) > 0:
            res_pr[i - 1] = np.nanmax(pr_list[mask])
            res_re[i - 1] = np.nanmean(re_list[mask])
        else:
            res_pr[i - 1] = np.nan
            res_re[i - 1] = np.nan
    return res_pr, res_re


if __name__ == "__main__":
    left_1, right_1, gt_1 = fetch_data("Country")

    pr_list = []
    re_list = []
    for analyzer, max_n_gram in product(["char", "char_wb"], [3, 4, 5]):
        if analyzer == "word" and max_n_gram > 2:
            continue
        precision, recall, f1 = fuzzy_join_precision_recall(
            left=left_1,
            right=right_1,
            gt=gt_1,
            left_col="title",
            right_col="title",
            analyzer=analyzer,
            ngram_range=(2, max_n_gram),
        )
        pr_list.extend(precision)
        re_list.extend(recall)
        # plt.plot(recall, precision,
        #         label=f"fuzzy_join_{analyzer}_{max_n_gram}_{similarity}")

    pr_list, re_list = best_precision_recall(np.array(pr_list), np.array(re_list))
    plt.plot(re_list, pr_list, label="fuzzy_join")

    pr_list_fw = []
    re_list_fw = []
    for scorer in partial_ratio, ratio, WRatio:
        precision_fw, recall_fw, f1_fw = thefuzz_precision_recall(
            left_1, right_1, gt_1, "title", "title", scorer=scorer
        )
        pr_list_fw.extend(precision_fw)
        re_list_fw.extend(recall_fw)

    pr_list_fw, re_list_fw = best_precision_recall(
        np.array(pr_list_fw), np.array(re_list_fw)
    )
    plt.plot(re_list_fw, pr_list_fw, label="thefuzz")
    # plt.plot(recall_fw, precision_fw, label=f'thefuzz_{scorer.__name__}')

    precision_fj, recall_fj, f1_fj = autofj_precision_recall(left_1, right_1, gt_1)
    plt.plot(recall_fj, precision_fj, label="autofj")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Best Precision / recall on Country")
    plt.savefig("precision_recall.png")
    # plt.show()
