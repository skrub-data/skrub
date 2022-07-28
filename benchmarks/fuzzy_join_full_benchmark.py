from fuzzy_join_benchmark import *

if __name__ == "__main__":
    datasets = [
        "Country",
        "BasketballTeam",
        "Drug",
        "Device",
        "ArtificialSatellite",
        "Amphibian",
        "Song",
        "HistoricBuilding",
        "Wrestler",
        "EthnicGroup",
    ]
    df = pd.DataFrame()
    for dataset in datasets:
        left_1, right_1, gt_1 = fetch_data(dataset)
        for analyser in ["word", "char", "char_wb"]:
            for max_n_gram in [3, 4]:
                for similarity in ["cosine", "l1", "l2"]:
                    if analyser == "word" and max_n_gram > 2:
                        continue
                    model_name = f"countVectorizer_{analyser}_{max_n_gram}_{similarity}"
                    precision, recall, f1 = FuzzyJoin_precision_recall(
                        right_1,
                        left_1,
                        gt_1,
                        "title",
                        "title",
                        analyzer=analyser,
                        ngram_range=(2, max_n_gram),
                        similarity=similarity,
                    )
                    n_points = len(precision)
                    df = df.append(
                        pd.DataFrame.from_dict(
                            {
                                "dataset": [dataset] * n_points,
                                "model": [model_name] * n_points,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                            },
                            orient="index",
                        ).transpose(),
                        ignore_index=True,
                    )

        fasttext.util.download_model("en", if_exists="ignore")  # English
        for similarity in ["cosine", "l1", "l2"]:
            model_name = f"fasttext_{similarity}"
            precision_ft, recall_ft, f1_ft = fasttext_precision_recall(
                right_1, left_1, gt_1, "title", "title", similarity=similarity
            )
            n_points = len(precision_ft)
            df = df.append(
                pd.DataFrame.from_dict(
                    {
                        "dataset": [dataset] * n_points,
                        "model": [model_name] * n_points,
                        "precision": precision_ft,
                        "recall": recall_ft,
                        "f1": f1_ft,
                    },
                    orient="index",
                ).transpose(),
                ignore_index=True,
            )

        for scorer in partial_ratio, ratio, WRatio:
            model_name = f"thefuzz_{scorer.__name__}"
            precision_fw, recall_fw, f1_fw = thefuzz_precision_recall(
                left_1, right_1, gt_1, "title", "title", scorer=scorer
            )
            n_points = len(precision_fw)
            df = df.append(
                pd.DataFrame.from_dict(
                    {
                        "dataset": [dataset] * n_points,
                        "model": [model_name] * n_points,
                        "precision": precision_fw,
                        "recall": recall_fw,
                        "f1": f1_fw,
                    },
                    orient="index",
                ).transpose(),
                ignore_index=True,
            )

        precision_fj, recall_fj, f1_fj = autofj_precision_recall(
            left_1, right_1, gt_1, n_points=10
        )
        model_name = "autofj_default"
        n_points = len(precision_fj)
        df = df.append(
            pd.DataFrame.from_dict(
                {
                    "dataset": [dataset] * n_points,
                    "model": [model_name] * n_points,
                    "precision": precision_fj,
                    "recall": recall_fj,
                    "f1": f1_fj,
                },
                orient="index",
            ).transpose(),
            ignore_index=True,
        )

    df.to_csv("full_benchmark_2.csv")
