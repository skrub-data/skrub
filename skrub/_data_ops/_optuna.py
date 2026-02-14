import contextlib
import itertools
import pathlib
import tempfile
import uuid

import joblib
import numpy as np
import sklearn
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state
from sklearn.utils.fixes import parse_version

from ._estimator import (
    _SEARCH_FITTED_ATTRIBUTES,
    _BaseParamSearch,
    _copy_attr,
    _get_classes,
    _SharedDict,
    _XyPipelineMixin,
    attribute_error,
)

_OPTUNA_SEARCH_FITTED_ATTRIBUTES = _SEARCH_FITTED_ATTRIBUTES + ["study_"]


def _parse_trial_params(trial):
    """
    Convert the keys and values we use for Optuna parameters, which are meant
    to be informative like {'1:estimator': '0:logistic', '0:C': 0.1} to the
    sklearn params like {'data_op__1': 0, 'data_op__0': 0.1}
    """
    return {
        f"data_op__{k.split(':', 1)[0]}": (
            int(v.split(":", 1)[0]) if isinstance(v, str) else v
        )
        for k, v in trial.params.items()
    }


def _process_trial_results(trial, cv_results, refit_metric):
    """
    Process results of one cross-validation and store them in the corresponding trial.

    refit_metric should be the name of the metric that drives the
    hyperparameter optimization (the return value of the objective function).
    If None, the first metric found in the CV results (ie the first one in the
    'scoring' passed to make_randomized_search) is used.
    """
    info = {}
    metrics = [
        c.removeprefix("test_") for c in cv_results.keys() if c.startswith("test_")
    ]
    for task in ("fit", "score"):
        info[f"mean_{task}_time"] = cv_results[f"{task}_time"].mean()
        info[f"std_{task}_time"] = cv_results[f"{task}_time"].std()
    for m in metrics:
        for task in ("train", "test"):
            try:
                scores = cv_results[f"{task}_{m}"]
            except KeyError:
                continue
            for split in range(len(scores)):
                info[f"split{split}_{task}_{m}"] = scores[split]
            info[f"mean_{task}_{m}"] = scores.mean()
            info[f"std_{task}_{m}"] = scores.std()
    info = {k: float(v) for k, v in info.items()}
    info["params"] = _parse_trial_params(trial)

    trial.set_user_attr("cv_results", info)
    if refit_metric is None:
        refit_metric = metrics[0]
    return info[f"mean_test_{refit_metric}"]


def _process_study_results(study):
    """
    Build the overall search results from the study once optimization is finished.
    """
    # Find all keys in the cv_results of all trials. Some trials may have
    # missing keys: when all splits raise an error only the 'params' key will
    # be present.
    all_keys = list(
        dict.fromkeys(
            itertools.chain(
                *(trial.user_attrs["cv_results"].keys() for trial in study.trials)
            )
        )
    )
    result = {k: [] for k in all_keys}
    # Build the overall cv_results_
    for trial in study.trials:
        trial_results = trial.user_attrs["cv_results"]
        for key, values in result.items():
            values.append(trial_results.get(key, None))
    return result


def _check_storage(url):
    """
    Convert URL to something we can pass to optuna.create_study

    SqlAlchemy URLs can be passed directly but when we want to use
    JournalStorage we need to instantiate it ourselves.
    """
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    if url.startswith("journal:///"):
        return JournalStorage(
            JournalFileBackend(file_path=url.removeprefix("journal:///"))
        )
    else:
        return url


def _get_scorer(estimator, scoring):
    """Create the scorer_ attribute."""
    if parse_version(sklearn.__version__) < parse_version("1.5"):
        if isinstance(scoring, (list, tuple, set)):
            scorer = {
                metric_name: check_scoring(estimator, metric_name)
                for metric_name in scoring
            }
        elif isinstance(scoring, dict):
            scorer = {k: check_scoring(estimator, v) for k, v in scoring.items()}
        else:
            scorer = check_scoring(estimator, scoring)
        return scorer
    scorer = check_scoring(estimator, scoring)
    try:
        # if multimetric, get the {name: scorer} dict
        scorer = scorer._scorers
    except AttributeError:
        pass
    return scorer


class OptunaParamSearch(_BaseParamSearch):
    """Learner that evaluates a skrub DataOp with hyperparameter tuning.

    This class is not meant to be instantiated manually, ``OptunaParamSearch``
    objects are created by calling
    :meth:`.skb.make_randomized_search(backend='optuna')
    <DataOp.skb.make_randomized_search>` on a :class:`DataOp`.

    Attributes of interest are ``best_learner_``, ``best_score_``,
    ``results_``, ``detailed_results_``, and ``study_`` (the Optuna study used
    to find hyperparameters).
    """

    def __init__(
        self,
        data_op,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
        # optuna-specific params
        storage=None,
        study_name=None,
        sampler=None,
        timeout=None,
    ):
        self.data_op = data_op
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.storage = storage
        self.study_name = study_name
        self.sampler = sampler
        self.timeout = timeout

    def __skrub_to_Xy_pipeline__(self, environment):
        new = _XyOptunaParamSearch(
            **self.get_params(deep=False), environment=_SharedDict(environment)
        )
        _copy_attr(self, new, _OPTUNA_SEARCH_FITTED_ATTRIBUTES)
        return new

    def fit(self, environment):
        import optuna
        import optuna.samplers

        self.scorer_ = _get_scorer(
            self.data_op.skb.make_learner().__skrub_to_Xy_pipeline__(environment),
            self.scoring,
        )
        self.refit_ = self.refit

        with contextlib.ExitStack() as exit_stack:
            #
            # Set up storage on disk so we can use multiprocessing
            #
            if self.storage is None:
                # We use a temp dir even if we just need one file because the
                # delete_on_close param of NamedTemporaryFile did not exist in
                # python < 3.12
                tmp_dir = exit_stack.enter_context(
                    tempfile.TemporaryDirectory(suffix="_skrub_optuna_search_storage")
                )
                tmp_file = pathlib.Path(tmp_dir) / "optuna_storage"
                storage = f"journal:///{tmp_file}"
            else:
                if not isinstance(self.storage, str):
                    raise TypeError(
                        f"storage should be a database URL or None, got: {self.storage}"
                    )
                storage = self.storage
            if self.study_name is None:
                study_name = f"skrub_randomized_search_{uuid.uuid4()}"
            else:
                study_name = self.study_name
            # Display storage as it should be passed to optuna-dashboard:
            # remove the protocol for journal files (only).
            print(
                f"Running optuna search for study {study_name} in storage "
                f"{storage.removeprefix('journal:///')}"
            )
            #
            # Create study and run trials
            #
            n_jobs = joblib.effective_n_jobs(self.n_jobs)
            random_state = check_random_state(self.random_state)
            seed = random_state.randint(np.iinfo("int32").max)
            sampler = (
                self.sampler
                if self.sampler is not None
                else optuna.samplers.TPESampler(seed=seed)
            )

            def create_study():
                # When using multiprocessing, create_study will get called by each
                # subprocess. Note we do not share a study nor an optuna
                # storage object with subprocesses, as they can store database
                # connections, thread pools etc., but create both the study
                # and the storage in each process.
                return optuna.create_study(
                    direction="maximize",
                    sampler=sampler,
                    storage=_check_storage(storage),
                    study_name=study_name,
                    load_if_exists=True,
                )

            def objective(trial):
                learner = self.data_op.skb.make_learner(choose=trial)
                try:
                    cv_results = learner.data_op.skb.cross_validate(
                        environment,
                        cv=self.cv,
                        error_score=self.error_score,
                        return_train_score=self.return_train_score,
                        verbose=self.verbose,
                        scoring=self.scoring,
                    )
                except Exception:
                    # Even with error_score != 'raise', cross_validate will
                    # raise if all splits raise an error.
                    if (
                        isinstance(self.error_score, str)
                        and self.error_score == "raise"
                    ):
                        raise
                    trial.set_user_attr(
                        "cv_results", {"params": _parse_trial_params(trial)}
                    )
                    return self.error_score
                refit_metric = self.refit_ if isinstance(self.refit_, str) else None
                return _process_trial_results(trial, cv_results, refit_metric)

            if (
                self.timeout is not None
                or getattr(
                    joblib.parallel.get_active_backend()[0], "uses_threads", False
                )
                or n_jobs == 1
            ):
                # If using timeout or sequential or threading parallelism, use
                # optuna's built-in parallelization
                study = create_study()
                study.optimize(
                    objective,
                    n_trials=self.n_iter,
                    n_jobs=n_jobs,
                    timeout=self.timeout,
                )
            else:
                # Otherwise for multiprocessing parallelism, use joblib.Parallel.

                # Make sure we initialize the database before launching the
                # processes otherwise we can have an error with rdb backends
                # where multiple processes try to create the same table (the
                # first one succeeds and the others get an error as the table
                # exists)
                create_study()

                def optimize():
                    study = create_study()
                    # reseed otherwise all processes will start with the same
                    # params, optuna also does this for each worker when
                    # n_jobs > 1
                    study.sampler.reseed_rng()
                    study.optimize(objective, n_trials=1, n_jobs=1)

                joblib.Parallel(n_jobs=n_jobs, pre_dispatch=self.pre_dispatch)(
                    joblib.delayed(optimize)() for _ in range(self.n_iter)
                )
            #
            # Copy results to an in-memory storage if the storage we used is a temp file
            #
            if self.storage is None:
                new_storage = optuna.storages.InMemoryStorage()
                optuna.study.copy_study(
                    from_study_name=study_name,
                    from_storage=_check_storage(storage),
                    to_storage=new_storage,
                    to_study_name=study_name,
                )
                self.study_ = optuna.study.load_study(
                    study_name=study_name, storage=new_storage, sampler=sampler
                )
            else:
                self.study_ = create_study()
            #
            # Best params & refit
            #
            self.cv_results_ = _process_study_results(self.study_)
            self.best_params_ = self.study_.best_params
            self.best_score_ = self.study_.best_value
            if not self.refit:
                return self
            best_learner = self.data_op.skb.make_learner(choose=self.study_.best_trial)
            best_learner.fit(environment)
            self.best_learner_ = best_learner
        return self


class _XyOptunaParamSearch(_XyPipelineMixin, OptunaParamSearch):
    def __init__(
        self,
        data_op,
        n_iter,
        scoring,
        n_jobs,
        refit,
        cv,
        verbose,
        pre_dispatch,
        random_state,
        error_score,
        return_train_score,
        storage,
        study_name,
        sampler,
        timeout,
        environment,
    ):
        self.data_op = data_op
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.storage = storage
        self.study_name = study_name
        self.sampler = sampler
        self.timeout = timeout
        self.environment = environment

    def __skrub_to_env_learner__(self):
        params = self.get_params(deep=False)
        params.pop("environment")
        new = OptunaParamSearch(**params)
        _copy_attr(self, new, _OPTUNA_SEARCH_FITTED_ATTRIBUTES)
        return new

    @property
    def classes_(self):
        if not hasattr(self, "best_learner_"):
            attribute_error(self, "classes_")
        try:
            return _get_classes(self.best_learner_.data_op)
        except AttributeError:
            attribute_error(self, "classes_")

    def fit(self, X, y=None):
        super().fit(self._get_env(X, y))
        return self

    def _call_predictor_method(self, name, X, y=None):
        return getattr(self.best_learner_, name)(self._get_env(X, y))
