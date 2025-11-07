import contextlib
import pathlib
import tempfile
import uuid
import warnings

import joblib
import numpy as np
import optuna
import optuna.samplers
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state

from ._estimator import (
    _SEARCH_FITTED_ATTRIBUTES,
    ParamSearch,
    _copy_attr,
    _get_classes,
    _SharedDict,
    _XyPipelineMixin,
    attribute_error,
)


def _get_metrics(cv_results):
    return [c.removeprefix("test_") for c in cv_results.keys() if c.startswith("test_")]


def _process_trial_results(trial, cv_results, refit_metric):
    info = {}
    metrics = _get_metrics(cv_results)
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
    info["params"] = {
        f"data_op__{k.split(':', 1)[0]}": (
            int(v.split(":", 1)[0]) if isinstance(v, str) else v
        )
        for k, v in trial.params.items()
    }
    trial.set_user_attr("cv_results", info)
    if refit_metric is None:
        refit_metric = metrics[0]
    return info[f"mean_test_{refit_metric}"]


def _process_study_results(study):
    result = {}
    for trial in study.trials:
        for k, v in trial.user_attrs["cv_results"].items():
            result.setdefault(k, []).append(v)
    return result


def _check_storage(url):
    if url.startswith("journal:"):
        return JournalStorage(
            JournalFileBackend(file_path=url.removeprefix("journal:"))
        )
    else:
        return url


class OptunaSearch(ParamSearch):
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

    def __skrub_to_Xy_pipeline__(self, environment):
        new = _XyOptunaSearch(
            **self.get_params(deep=False), environment=_SharedDict(environment)
        )
        _copy_attr(self, new, _SEARCH_FITTED_ATTRIBUTES + ["study_"])
        return new

    def fit(self, environment):
        scorer = check_scoring(
            self.data_op.skb.make_learner().__skrub_to_Xy_pipeline__(environment),
            self.scoring,
        )
        try:
            self.scorer_ = scorer._scorers
        except AttributeError:
            self.scorer_ = scorer
        self.refit_ = self.refit

        with contextlib.ExitStack() as exit_stack:
            #
            # Set up storage on disk so we can use multiprocessing
            #
            if self.storage is None:
                tmp_file_obj = exit_stack.enter_context(
                    tempfile.NamedTemporaryFile(delete_on_close=False)
                )
                tmp_file = tmp_file_obj.name
                tmp_file_obj.close()
                storage = f"journal:{tmp_file}"
            else:
                if not isinstance(self.storage, (str, pathlib.Path)):
                    raise TypeError(
                        f"storage should be a database url or None, got: {self.storage}"
                    )
            if self.study_name is None:
                study_name = f"skrub_randomized_search_{uuid.uuid4()}"
            else:
                study_name = self.study_name
            #
            # Create study and run trials
            #
            n_jobs = joblib.effective_n_jobs(self.n_jobs)
            if n_jobs == 1:
                random_state = check_random_state(self.random_state)
                seed = random_state.randint(np.iinfo("int32").max)
            else:
                if self.random_state is not None:
                    warnings.warn(
                        "Optuna search with n_jobs > 1 is not deterministic, setting"
                        " random_state to None"
                    )
                seed = None

            sampler = (
                self.sampler
                if self.sampler is not None
                else optuna.samplers.TPESampler(seed=seed)
            )

            def create_study():
                return optuna.create_study(
                    direction="maximize",
                    sampler=sampler,
                    storage=_check_storage(storage),
                    study_name=study_name,
                    load_if_exists=True,
                )

            def objective(trial):
                learner = self.data_op.skb.make_learner(choose=trial)
                cv_results = learner.data_op.skb.cross_validate(
                    environment,
                    cv=self.cv,
                    error_score=self.error_score,
                    return_train_score=self.return_train_score,
                    verbose=self.verbose,
                    scoring=self.scoring,
                )
                return _process_trial_results(trial, cv_results, None)

            if (
                getattr(joblib.parallel.get_active_backend()[0], "uses_threads", False)
                or n_jobs == 1
            ):
                # If sequential or threading, use optuna's built-in parallelization
                study = create_study()
                study.optimize(objective, n_trials=self.n_iter, n_jobs=n_jobs)
            else:
                # Otherwise use joblib.Parallel
                # Note this would probably not be safe with the threading
                # backend.
                def optimize():
                    study = create_study()
                    study.optimize(objective, n_trials=1, n_jobs=1)

                joblib.Parallel(n_jobs=n_jobs, pre_dispatch=self.pre_dispatch)(
                    joblib.delayed(optimize)() for _ in range(self.n_iter)
                )
            #
            # Copy results to an in-memory storage if self.storage was None
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
                self.study_ = study
            #
            # Best params & refit
            #
            self.cv_results_ = _process_study_results(self.study_)
            self.best_params_ = self.study_.best_params
            if not self.refit:
                return self
            best_learner = self.data_op.skb.make_learner(choose=self.study_.best_trial)
            best_learner.fit(environment)
            self.best_learner_ = best_learner
        return self


class _XyOptunaSearch(_XyPipelineMixin, OptunaSearch):
    def __init__(
        self,
        data_op,
        n_iter,
        scoring,
        n_jobs,
        refit,
        cv,
        random_state,
        storage,
        study_name,
        environment,
    ):
        self.data_op = data_op
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.random_state = random_state
        self.storage = storage
        self.study_name = study_name
        self.environment = environment

    def __skrub_to_env_learner__(self):
        new = OptunaSearch(**self.get_params(deep=False))
        _copy_attr(self, new, _SEARCH_FITTED_ATTRIBUTES + ["study_"])
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
