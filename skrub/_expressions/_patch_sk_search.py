import time
from collections import defaultdict
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics._scorer import (
    _MultimetricScorer,
)
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _fit_and_score,
    _insert_error_scores,
    _warn_or_raise_about_fit_failures,
)
from sklearn.utils.validation import _check_method_params, indexable


def progress_bar(
    idx,
    total,
    score_name,
    best_score,
    best_parameters,
    bar_length=30,
):
    fraction = idx / total
    filled_length = int(round(bar_length * fraction))

    # Construct the bar with an orange filled part and a default unfilled part
    orange = "\033[38;5;208m"  # ANSI code for orange
    reset = "\033[0m"  # Reset color
    bar = f"{orange}{'█' * filled_length}{reset}{'-' * (bar_length - filled_length)}"

    # Print the progress bar on the same line using carriage return (\r)
    text = " | ".join(
        [
            f"\r[ {bar} ] {int(fraction * 100)}%",
            f"Best test {score_name}: {best_score}",
            f"Best params: {best_parameters}",
        ]
    )
    print(text, end="", flush=True)


def _get_score_name(scorers):
    scorers = getattr(scorers, "_scorer", scorers)

    if hasattr(scorers, "_score_func"):
        return scorers._score_func.__name__

    if isinstance(est := getattr(scorers, "_estimator", None), BaseEstimator):
        return {"regressor": "r2", "classifier": "accuracy"}[est._estimator_type]

    return "score"


def _fit(self, X, y=None, **params):
    """Run fit with all sets of parameters.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features. For precomputed kernel or
        distance matrix, the expected shape of X is (n_samples, n_samples).

    y : array-like of shape (n_samples, n_output) \
        or (n_samples,), default=None
        Target relative to X for classification or regression;
        None for unsupervised learning.

    **params : dict of str -> object
        Parameters passed to the ``fit`` method of the estimator, the scorer,
        and the CV splitter.

        If a fit parameter is an array-like whose length is equal to
        `num_samples` then it will be split by cross-validation along with
        `X` and `y`. For example, the :term:`sample_weight` parameter is
        split because `len(sample_weights) = len(X)`. However, this behavior
        does not apply to `groups` which is passed to the splitter configured
        via the `cv` parameter of the constructor. Thus, `groups` is used
        *to perform the split* and determines which samples are
        assigned to the each side of the a split.

    Returns
    -------
    self : object
        Instance of fitted estimator.
    """
    estimator = self.estimator
    scorers, refit_metric = self._get_scorers()

    X, y = indexable(X, y)
    params = _check_method_params(X, params=params)

    routed_params = self._get_routed_params_for_fit(params)

    cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
    n_splits = cv_orig.get_n_splits(X, y, **routed_params.splitter.split)

    base_estimator = clone(self.estimator)

    parallel = Parallel(
        n_jobs=self.n_jobs,
        pre_dispatch=self.pre_dispatch,
        return_as="generator",
    )

    fit_and_score_kwargs = dict(
        scorer=scorers,
        fit_params=routed_params.estimator.fit,
        score_params=routed_params.scorer.score,
        return_train_score=self.return_train_score,
        return_n_test_samples=True,
        return_times=True,
        return_parameters=True,
        error_score=self.error_score,
        verbose=self.verbose,
    )
    results = {}
    with parallel:
        all_candidate_params = []
        all_out = []
        all_more_results = defaultdict(list)

        def evaluate_candidates(candidate_params, cv=None, more_results=None):
            cv = cv or cv_orig
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            if self.verbose > 0:
                print(
                    "Fitting {0} folds for each of {1} candidates,"
                    " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits
                    )
                )

            out_gen = parallel(
                delayed(_fit_and_score)(
                    clone(base_estimator),
                    X,
                    y,
                    train=train,
                    test=test,
                    parameters=parameters,
                    split_progress=(split_idx, n_splits),
                    candidate_progress=(cand_idx, n_candidates),
                    **fit_and_score_kwargs,
                )
                for (cand_idx, parameters), (split_idx, (train, test)) in product(
                    enumerate(candidate_params),
                    enumerate(cv.split(X, y, **routed_params.splitter.split)),
                )
            )

            score_name = _get_score_name(scorers)

            best_score = -float("inf")
            best_score_str = ""
            best_parameters = {}
            score_buffer = []
            parameters_buffer = []
            out = []

            progress_bar(
                0,
                n_candidates,
                score_name,
                best_score_str,
                best_parameters,
            )

            for idx, result in enumerate(out_gen, 1):
                out.append(result)
                parameters_buffer.append(result["parameters"])
                score_buffer.append(result["test_scores"])

                if idx % n_splits == 0:
                    # We evaluated all folds of a single parameters combination.
                    # Parallel(return_as = generator) sequence is ordered.
                    assert all([parameters_buffer[0] == p for p in parameters_buffer])
                    mean_score = np.mean(score_buffer)

                    # Higher is better by sklearn convention
                    if mean_score > best_score:
                        best_score = mean_score
                        best_parameters = parameters_buffer[0]

                        std_score = np.std(score_buffer)
                        best_score_str = f"{mean_score:.4f} ± {std_score:.4f}"

                    progress_bar(
                        idx // n_splits,
                        n_candidates,
                        score_name,
                        best_score_str,
                        best_parameters,
                    )
                    parameters_buffer = []
                    score_buffer = []

            print()  # Move to the next line when done

            if len(out) < 1:
                raise ValueError(
                    "No fits were performed. "
                    "Was the CV iterator empty? "
                    "Were there no candidates?"
                )
            elif len(out) != n_candidates * n_splits:
                raise ValueError(
                    "cv.split and cv.get_n_splits returned "
                    "inconsistent results. Expected {} "
                    "splits, got {}".format(n_splits, len(out) // n_candidates)
                )

            _warn_or_raise_about_fit_failures(out, self.error_score)

            # For callable self.scoring, the return type is only know after
            # calling. If the return type is a dictionary, the error scores
            # can now be inserted with the correct key. The type checking
            # of out will be done in `_insert_error_scores`.
            if callable(self.scoring):
                _insert_error_scores(out, self.error_score)

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

            if more_results is not None:
                for key, value in more_results.items():
                    all_more_results[key].extend(value)

            nonlocal results
            results = self._format_results(
                all_candidate_params, n_splits, all_out, all_more_results
            )

            return results

        self._run_search(evaluate_candidates)

        # multimetric is determined here because in the case of a callable
        # self.scoring the return type is only known after calling
        first_test_score = all_out[0]["test_scores"]
        self.multimetric_ = isinstance(first_test_score, dict)

        # check refit_metric now for a callable scorer that is multimetric
        if callable(self.scoring) and self.multimetric_:
            self._check_refit_for_multimetric(first_test_score)
            refit_metric = self.refit

    # For multi-metric evaluation, store the best_index_, best_params_ and
    # best_score_ iff refit is one of the scorer names
    # In single metric evaluation, refit_metric is "score"
    if self.refit or not self.multimetric_:
        self.best_index_ = self._select_best_index(self.refit, refit_metric, results)
        if not callable(self.refit):
            # With a non-custom callable, we can select the best score
            # based on the best index
            self.best_score_ = results[f"mean_test_{refit_metric}"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

    if self.refit:
        # here we clone the estimator as well as the parameters, since
        # sometimes the parameters themselves might be estimators, e.g.
        # when we search over different estimators in a pipeline.
        # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
        self.best_estimator_ = clone(base_estimator).set_params(
            **clone(self.best_params_, safe=False)
        )

        refit_start_time = time.time()
        if y is not None:
            self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
        else:
            self.best_estimator_.fit(X, **routed_params.estimator.fit)
        refit_end_time = time.time()
        self.refit_time_ = refit_end_time - refit_start_time

        if hasattr(self.best_estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.best_estimator_.feature_names_in_

    # Store the only scorer not as a dict for single metric evaluation
    if isinstance(scorers, _MultimetricScorer):
        self.scorer_ = scorers._scorers
    else:
        self.scorer_ = scorers

    self.cv_results_ = results
    self.n_splits_ = n_splits

    return self


BaseSearchCV.fit = _fit
