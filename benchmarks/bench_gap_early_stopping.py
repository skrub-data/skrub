"""
This benchmark compares using `GapEncoder` with and without early-stopping.

The logic is as follows:
- Modify the GapEncoder to implement early-stopping and expose its parameters.
- Tweak the default parameters of the Gap:
  - Set RNG seed
  - Set `min_iter` to 5 (`= max_iter`)
  - Set `max_iter_e_step` to 2 (from the benchmark in
    https://github.com/skrub-data/skrub/pull/593 final performance does not depend much on this parameter)
- At each iteration, save the following:
  - `score`, from ``self.score(X)``
- Run benchmark comparing
    1. Gap without early-stopping
    2. Gap with early-stopping
    3. Gap with early_stopping with update every 5 monitor_steps

Date: July 20th 2023
Commit: TODO
"""

import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

from argparse import ArgumentParser
from skrub._gap_encoder import (
    GapEncoder,
    GapEncoderColumn,
    _multiplicative_update_h,
    _multiplicative_update_w,
    batch_lookup,
    check_input,
)
from joblib import Parallel, delayed
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from skrub import TableVectorizer
from pathlib import Path

from utils import (
    monitor,
    default_parser,
    find_result,
    get_classification_datasets,
    get_regression_datasets,
)


class ModifiedGapEncoderColumn(GapEncoderColumn):
    def __init__(
        self,
        *args,
        column_name: str = "MISSING COLUMN",
        monitor_steps=10,
        early_stop=False,
        patience=5,
        min_delta_score=0.001,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.column_name = column_name
        self.benchmark_results_: list[dict[str, np.ndarray | float]] = []
        self.monitor_steps = monitor_steps
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta_score = min_delta_score

    def fit(self, X, y=None):
        # Copy parameter rho
        self.rho_ = self.rho
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "Input data is not string. "
        # Make n-grams counts matrix unq_V
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
        # Get activations unq_H
        unq_H = self._get_H(unq_X)

        n_iter_patience = 0
        min_score = None
        stop = False
        for n_iter_ in range(self.max_iter):
            previous_steps = n_batch * n_iter_
            # Loop over batches
            for i, (unq_idx, idx) in enumerate(batch_lookup(lookup, n=self.batch_size)):
                # if i == n_batch - 1:
                if i % self.monitor_steps == 0:
                    W_last = self.W_.copy()
                # Update activations unq_H
                unq_H[unq_idx] = _multiplicative_update_h(
                    unq_V[unq_idx],
                    self.W_,
                    unq_H[unq_idx],
                    epsilon=1e-3,
                    max_iter=self.max_iter_e_step,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior,
                )
                # Update the topics self.W_
                _multiplicative_update_w(
                    unq_V[idx],
                    self.W_,
                    self.A_,
                    self.B_,
                    unq_H[idx],
                    self.rescale_W,
                    self.rho_,
                )

                # if i == n_batch - 1:
                #    # Compute the norm of the update of W in the last batch
                if i % self.monitor_steps == 0:
                    W_change = np.linalg.norm(self.W_ - W_last) / np.linalg.norm(W_last)

                    score = self.score(X)

                    self.benchmark_results_.append(
                        {
                            "column_name": self.column_name,
                            "score": score,
                            "n_batch": i,
                            "n_iter": n_iter_,
                            "global_gap_iter": i + previous_steps,
                            "n_batches": n_batch,
                        }
                    )

                    if self.early_stop:
                        if min_score is None:
                            min_score = score
                        # elif score < min_score:
                        elif (min_score - score) / min_score > self.min_delta_score:
                            if min_score - score > 0:
                                min_score = score
                            n_iter_patience = 0
                        else:
                            n_iter_patience += 1
                            if n_iter_patience > self.patience:
                                stop = True
                                break

            # score = self.score(X)

            if stop or ((W_change < self.tol) and (n_iter_ >= self.min_iter - 1)):
                break  # Stop if the change in W is smaller than the tolerance

        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
        return self


class ModifiedGapEncoder(GapEncoder):
    fitted_models_: list[ModifiedGapEncoderColumn]

    def __init__(self, monitor_steps=10, early_stop=False, patience=5, **kwargs):
        super().__init__(**kwargs)
        self.monitor_steps = monitor_steps
        self.early_stop = early_stop
        self.patience = patience

    def _create_column_gap_encoder(self, column_name: str):
        return ModifiedGapEncoderColumn(
            column_name=column_name,
            ngram_range=self.ngram_range,
            n_components=self.n_components,
            analyzer=self.analyzer,
            gamma_shape_prior=self.gamma_shape_prior,
            gamma_scale_prior=self.gamma_scale_prior,
            rho=self.rho,
            rescale_rho=self.rescale_rho,
            batch_size=self.batch_size,
            tol=self.tol,
            hashing=self.hashing,
            hashing_n_features=self.hashing_n_features,
            max_iter=self.max_iter,
            init=self.init,
            add_words=self.add_words,
            random_state=self.random_state,
            rescale_W=self.rescale_W,
            max_iter_e_step=self.max_iter_e_step,
            monitor_steps=self.monitor_steps,
            early_stop=self.early_stop,
            patience=self.patience,
        )

    def fit(self, X, y=None):
        # Check that n_samples >= n_components
        if len(X) < self.n_components:
            raise ValueError(
                f"n_samples={len(X)} should be >= n_components={self.n_components}. "
            )
        # Copy parameter rho
        self.rho_ = self.rho
        # If X is a dataframe, store its column names
        if isinstance(X, pd.DataFrame):
            self.column_names_ = list(X.columns)
        # Check input data shape
        X = check_input(X)
        X = self._handle_missing(X)
        self.fitted_models_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._create_column_gap_encoder(self.column_names_[k]).fit)(X[:, k])
            for k in range(X.shape[1])
        )
        return self


benchmark_name = Path(__file__).stem


@monitor(
    parametrize={
        "early_stop_config": [
            {"early_stop": False, "patience": -1, "monitor_steps": 1},
            {"early_stop": True, "patience": 5, "monitor_steps": 1},
            {"early_stop": True, "patience": 1, "monitor_steps": 5},
        ],
        "dataset_name": [
            "medical_charge",
            "open_payments",
            "midwest_survey",
            "employee_salaries",
            # "road_safety",  # https://github.com/skrub-data/skrub/issues/622
            "drug_directory",
            "traffic_violations",
        ],
        "batch_size": [
            32,
            128,
            512,
        ],
        "max_iter_e_step": [2, 5, 10],
    },
    save_as=benchmark_name,
    repeat=3,
)
def benchmark(
    early_stop_config: list, dataset_name: str, batch_size: int, max_iter_e_step: int
):
    """
    Cross-validate a pipeline with a modified `GapEncoder` instance for the
    high cardinality column. The rest of the columns are dropped.
    """

    max_rows = 40_000

    dataset = dataset_map[dataset_name]

    if dataset_name in regression_datasets:
        estimator = HistGradientBoostingRegressor(random_state=0)
    elif dataset_name in classification_datasets:
        estimator = HistGradientBoostingClassifier(random_state=0)

    cv = cross_validate(
        Pipeline(
            [
                (
                    "encoding",
                    TableVectorizer(
                        high_card_cat_transformer=ModifiedGapEncoder(
                            min_iter=2,
                            max_iter=5,
                            random_state=0,
                            early_stop=early_stop_config["early_stop"],
                            patience=early_stop_config["patience"],
                            monitor_steps=early_stop_config["monitor_steps"],
                            batch_size=batch_size,
                            max_iter_e_step=max_iter_e_step,
                        ),
                        # drop all other features to check possible degradation in prediction
                        # due to early-stopping in gap-encoded features
                        low_card_cat_transformer="drop",
                        numerical_transformer="drop",
                        datetime_transformer="drop",
                        remainder="drop",
                        n_jobs=-1,
                    ),
                ),
                ("model", estimator),
            ]
        ),
        dataset.X[:max_rows],
        dataset.y[:max_rows],
        return_estimator=True,
        n_jobs=-1,
        cv=3,
    )

    # Extract the estimators from the cross-validation results
    pipelines = cv.pop("estimator")
    # Transform the rest to a DataFrame, which will be easier to iterative over
    cv_df = pd.DataFrame(cv)

    results = []
    for pipeline, (_, cv_results) in zip(pipelines, cv_df.iterrows()):
        for modified_gap_encoder in (
            pipeline["encoding"].named_transformers_["high_card_cat"].fitted_models_
        ):
            for gap_iter, inner_results in enumerate(
                modified_gap_encoder.benchmark_results_
            ):
                cardinality = len(dataset.X[inner_results["column_name"]].unique())
                entropy = sp.stats.entropy(
                    dataset.X[inner_results["column_name"]].value_counts(
                        normalize=True, sort=False
                    )
                )

                loop_results = {
                    "dataset": dataset_name,
                    "column_name": f"{dataset_name}.{inner_results['column_name']}",
                    "cv_test_score": cv_results["test_score"],
                    "gap_iter": gap_iter
                    + 1,  # counts the results recorded, so max_iter * n_batches if no early-stopping
                    "n_batch": inner_results[
                        "n_batch"
                    ],  # counts the number of batches processed
                    "n_iter": inner_results[
                        "n_iter"
                    ],  # counts the external iterations, from min_iter to max_iter
                    "global_gap_iter": inner_results[
                        "global_gap_iter"
                    ],  # needed when monitor_steps > 1
                    "score": inner_results["score"],
                    "cardinality": cardinality,
                    "entropy": entropy,
                    "n_batches": inner_results["n_batches"],
                }
                results.append(loop_results)

    # transform a list of dicts to a dict of lists
    results = {key: [dic[key] for dic in results] for key in results[0]}
    return results


def plot(df: pd.DataFrame):
    for config_str in df["early_stop_config"].unique().tolist():
        config = eval(config_str)
        if config["early_stop"] == False:
            results_df = df[df["early_stop_config"] == config_str]
        elif config["monitor_steps"] == 1:
            results_df_early_stop = df[df["early_stop_config"] == config_str]
        else:
            results_df_early_stop_skip = df[df["early_stop_config"] == config_str]

    sns.set_theme(style="ticks", palette="pastel")

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    n_batches = results_df["n_batches"].max()

    hue_order = results_df["column_name"].unique().tolist()

    sns.lineplot(
        data=results_df,
        x="global_gap_iter",
        y="score",
        hue="column_name",
        hue_order=hue_order,
        legend=True,
        linewidth=1,
        ax=ax1,
    )

    sns.lineplot(
        data=results_df_early_stop,
        x="global_gap_iter",
        y="score",
        hue="column_name",
        hue_order=hue_order,
        legend=False,
        marker="o",
        markeredgecolor="r",
        markerfacecolor="none",
        markeredgewidth=1,
        markersize=8,
        linewidth=1,
        linestyle="-.",
        ax=ax1,
    )

    sns.lineplot(
        data=results_df_early_stop_skip,
        x="global_gap_iter",
        y="score",
        hue="column_name",
        hue_order=hue_order,
        legend=False,
        marker="*",
        markerfacecolor="r",
        markeredgecolor="k",
        markeredgewidth=1,
        markersize=6,
        linewidth=1,
        linestyle="--",
        ax=ax1,
    )

    ax1.set_yscale("log")
    ax1.set_xlim(xmax=n_batches * (results_df["n_iter"].max() + 1))

    for i in range(1, results_df["n_iter"].astype(int).max() + 1):
        ax1.axvline(x=n_batches * i, color="gray", linestyle=":")

    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.41))

    ax2.bar(
        ["No Early-Stop", "Early-Stop", "Early-Stop Skip Steps"],
        [
            results_df.gap_iter.max(),
            results_df_early_stop.gap_iter.max(),
            results_df_early_stop_skip.gap_iter.max(),
        ],
        yerr=[
            results_df.gap_iter.std(),
            results_df_early_stop.gap_iter.std(),
            results_df_early_stop_skip.gap_iter.std(),
        ],
    )
    ax2.set_ylabel("total n. of batches processed")

    ax3.bar(
        ["No Early-Stop", "Early-Stop", "Early-Stop Skip Steps"],
        [
            results_df.cv_test_score.mean(),
            results_df_early_stop.cv_test_score.mean(),
            results_df_early_stop_skip.cv_test_score.mean(),
        ],
        yerr=[
            results_df.cv_test_score.std(),
            results_df_early_stop.cv_test_score.std(),
            results_df_early_stop_skip.cv_test_score.std(),
        ],
    )
    ax3.set_ylabel("cv test score")

    results_df = results_df.merge(
        results_df_early_stop,
        how="left",
        on=["global_gap_iter", "column_name", "n_iter"],
        suffixes=("", "_ES"),
    )
    results_df = results_df.merge(
        results_df_early_stop_skip,
        how="left",
        on=["global_gap_iter", "column_name", "n_iter"],
        suffixes=("", "_ES_skip"),
    )

    results_df["relative_score_diff_early-stop"] = (
        results_df["score"] - results_df["score_ES"]
    ).abs() / results_df["score"]
    results_df["relative_score_diff_early-stop_skip"] = (
        results_df["score"] - results_df["score_ES_skip"]
    ).abs() / results_df["score"]

    sns.lineplot(
        data=results_df,
        x="global_gap_iter",
        y="relative_score_diff_early-stop",
        hue="column_name",
        hue_order=hue_order,
        legend=False,
        markersize=5,
        ax=ax4,
    )

    sns.lineplot(
        data=results_df,
        x="global_gap_iter",
        y="relative_score_diff_early-stop_skip",
        hue="column_name",
        hue_order=hue_order,
        legend=False,
        linestyle="--",
        markersize=5,
        ax=ax4,
    )

    # ax4.set_yscale('log')
    ax4.set_xlim(xmax=n_batches * (results_df["n_iter"].max() + 1))

    for i in range(1, results_df["n_iter"].astype(int).max() + 1):
        ax4.axvline(x=n_batches * i, color="gray", linestyle=":")

    sns.stripplot(
        data=results_df,
        x="cardinality",
        y="score",
        hue="column_name",
        hue_order=hue_order,
        legend=False,
        ax=ax5,
    )
    ax5.set_yscale("log")

    sns.stripplot(
        data=results_df,
        x="entropy",
        y="score",
        hue="column_name",
        hue_order=hue_order,
        legend=False,
        ax=ax6,
    )
    ax6.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:,.2f}".format(x))
    )
    ax6.set_yscale("log")

    plt.show()


if __name__ == "__main__":
    _args = ArgumentParser(
        description="Benchmark for the GapEncoder early-stopping",
        parents=[default_parser],
    ).parse_args()

    if _args.run:
        regression_datasets = get_regression_datasets()
        classification_datasets = get_classification_datasets()
        dataset_map = {
            key: value for value, key in regression_datasets + classification_datasets
        }
        regression_datasets = [key for value, key in regression_datasets]
        classification_datasets = [key for value, key in classification_datasets]

        df = benchmark()
    else:
        result_file = find_result(benchmark_name)
        df = pd.read_parquet(result_file)

    if _args.plot:
        plot(df)
