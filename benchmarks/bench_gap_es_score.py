"""
Benchmark hyperparameters of GapEncoder on traffic_violations dataset
"""

from utils import default_parser, find_result, monitor
from time import perf_counter
import numpy as np
import pandas as pd
from skrub.datasets import fetch_traffic_violations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from skrub import GapEncoder
from skrub._gap_encoder import (
    GapEncoderColumn,
    _beta_divergence,
    batch_lookup,
    _multiplicative_update_h,
    _multiplicative_update_w,
)
import seaborn as sns
import matplotlib.pyplot as plt


class ModifiedGapEncoderColumn(GapEncoderColumn):
    def __init__(self, *args, **kwargs):
        if "max_no_improvement" in kwargs:
            self.max_no_improvement = kwargs.pop("max_no_improvement")
        if "verbose" in kwargs:
            self.verbose = kwargs.pop("verbose")
        super().__init__(*args, **kwargs)

    def _minibatch_convergence(self, batch_size, batch_cost, n_samples, step, n_steps):
        """Helper function to encapsulate the early stopping logic"""
        # adapted from sklearn.decomposition.MiniBatchNMF

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because H is not updated yet.
        if step == 1:
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}")
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True

        return False

    def fit(self, X, y=None) -> "GapEncoderColumn":
        """
        Fit the GapEncoder on `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        GapEncoderColumn
            The fitted GapEncoderColumn instance (self).
        """
        # Copy parameter rho
        self.rho_ = self.rho
        # Attributes to monitor the convergence
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "Input data is not string. "
        # Make n-grams counts matrix unq_V
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
        n_samples = len(X)
        del X
        # Get activations unq_H
        unq_H = self._get_H(unq_X)

        for n_iter_ in range(self.max_iter):
            # Loop over batches
            for i, (unq_idx, idx) in enumerate(batch_lookup(lookup, n=self.batch_size)):
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
                batch_cost = _beta_divergence(
                    unq_V[idx],
                    unq_H[idx],
                    self.W_,
                    "kullback-leibler",
                    square_root=False,
                )
                if self._minibatch_convergence(
                    batch_size=len(idx),
                    batch_cost=batch_cost,
                    n_samples=n_samples,
                    step=n_iter_ * n_batch + i,
                    n_steps=self.max_iter * n_batch,
                ):
                    break
            else:
                # only continue if no break occurred
                continue
            break

        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
        return self


class ModifiedGapEncoder(GapEncoder):
    fitted_models_: list[ModifiedGapEncoderColumn]

    def _create_column_gap_encoder(self):
        return ModifiedGapEncoderColumn(
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
            max_no_improvement=10,
            verbose=True,
        )


###############################################################
# Benchmarking accuracy and speed on traffic_violations dataset
###############################################################

benchmark_name = "gap_encoder_benchmark_es_score"


@monitor(
    memory=True,
    time=True,
    parametrize={
        "high_card_feature": [
            "seqid",
            "description",
            "location",
            "search_reason_for_stop",
            "state",
            "charge",
            "driver_city",
            "driver_state",
            "dl_state",
        ],
        "max_rows": [5_000, 20_000, 50_000],
        "modif": [True, False],
    },
    save_as=benchmark_name,
    repeat=2,
)
def benchmark(
    high_card_feature: str,
    max_rows: int,
    modif: bool,
):
    ds = fetch_traffic_violations()
    X = np.array(ds.X[high_card_feature]).reshape(-1, 1).astype(str)
    y = ds.y
    # only keep the first max_rows rows
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X[:max_rows],
        y[:max_rows],
        test_size=0.2,
    )
    if not modif:
        gap = GapEncoder(batch_size=512)
    else:
        gap = ModifiedGapEncoder(verbose=False, batch_size=512)
    start_time = perf_counter()
    gap.fit(X_train)
    end_time = perf_counter()
    score_train = gap.score(X_train)
    score_test = gap.score(X_test)

    # evaluate the accuracy using the encoding
    X_train_encoded = gap.transform(X_train)
    X_test_encoded = gap.transform(X_test)

    clf = HistGradientBoostingClassifier()
    clf.fit(X_train_encoded, y_train)
    roc_auc_hgb_train = roc_auc_score(
        y_train, clf.predict_proba(X_train_encoded), multi_class="ovr"
    )
    roc_auc_hgb_test = roc_auc_score(
        y_test, clf.predict_proba(X_test_encoded), multi_class="ovr"
    )
    balanced_accuracy_hgb_train = balanced_accuracy_score(
        y_train, clf.predict(X_train_encoded)
    )
    balanced_accuracy_hgb_test = balanced_accuracy_score(
        y_test, clf.predict(X_test_encoded)
    )

    res_dic = {
        "time_fit": end_time - start_time,
        "score_train": score_train,
        "score_test": score_test,
        "roc_auc_hgb_train": roc_auc_hgb_train,
        "roc_auc_hgb_test": roc_auc_hgb_test,
        "balanced_accuracy_hgb_train": balanced_accuracy_hgb_train,
        "balanced_accuracy_hgb_test": balanced_accuracy_hgb_test,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
    }

    return res_dic


def plot(df: pd.DataFrame):
    sns.lineplot(
        x="train_size", y="time_fit", data=df, hue="high_card_feature", style="modif"
    )
    plt.yscale("log")
    # put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.ylabel("Time (s)")
    plt.xlabel("Train size")
    plt.title("Time to fit the encoder")
    # make sure the plot is not cut
    plt.tight_layout()
    plt.show()

    sns.lineplot(
        x="train_size", y="score_train", data=df, hue="high_card_feature", style="modif"
    )
    plt.yscale("log")
    # put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.ylabel("Score")
    plt.xlabel("Train size")
    plt.title("Score on train set")
    # make sure the plot is not cut
    plt.tight_layout()
    plt.show()

    sns.lineplot(
        x="train_size",
        y="balanced_accuracy_hgb_test",
        data=df,
        hue="high_card_feature",
        style="modif",
    )
    plt.yscale("log")
    # put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.ylabel("Balanced accuracy")
    plt.xlabel("Train size")
    plt.title("Balanced accuracy on test set")
    # make sure the plot is not cut
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    _args = ArgumentParser(
        description="Benchmark for the batch feature of the MinHashEncoder.",
        parents=[default_parser],
    ).parse_args()

    if _args.run:
        df = benchmark()
    else:
        result_file = find_result(benchmark_name)
        df = pd.read_parquet(result_file)

    if _args.plot:
        plot(df)
