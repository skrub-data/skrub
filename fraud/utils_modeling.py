# ruff: noqa
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import TunedThresholdClassifierCV


def _get_palette(names):
    return dict(zip(names, sns.color_palette("colorblind", n_colors=len(names))))


def plot_metric(results, metric_name):
    """Plot a bar graph comparing all models for the desired metric."""
    values = []
    for named_results in results.values():
        values.append(named_results[metric_name])

    names = list(results)
    palette = _get_palette(names)

    fig, ax = plt.subplots()
    ax = sns.barplot(y=names, x=values, hue=names, palette=palette.values(), orient="h")
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title(metric_name)
    plt.tight_layout()


def plot_gains_curve(results):
    """Plot a gain curve obtained after using TunedThresholdClassifier."""
    tuned_models = {
        name: named_results["model"]
        for name, named_results in results.items()
        if isinstance(named_results["model"], TunedThresholdClassifierCV)
        and named_results["model"].store_cv_results
    }
    if len(tuned_models) == 0:
        return

    names = list(results)
    palette = _get_palette(names)

    fig, ax = plt.subplots()

    for name, model in tuned_models.items():
        ax.plot(
            model.cv_results_["thresholds"],
            model.cv_results_["scores"],
            color=palette[name],
        )
        ax.plot(
            model.best_threshold_,
            model.best_score_,
            "o",
            markersize=10,
            color=palette[name],
            label=f"Optimal cut-off point for the business metric of {name}",
        )
        ax.legend()
        ax.set_xlabel("Decision threshold (probability)")
        ax.set_ylabel("Objective score (using cost-matrix)")
        ax.set_title("Objective score as a function of the decision threshold")


def plot_roc_curve(results):
    names = list(results)
    palette = _get_palette(names)

    fig, ax = plt.subplots()
    for idx, (name, named_results) in enumerate(results.items()):
        y_proba = named_results["y_proba"][:, 1]
        y_test = named_results["y_test"]
        model = named_results["model"]

        ax.plot(
            named_results["fpr"],
            named_results["recall"],
            marker="o",
            color=palette[name],
            markersize=10,
            label=f"Cut-off point for {named_results['threshold']:.3f}",
        )

        is_last_plot = idx == len(results) - 1

        RocCurveDisplay.from_predictions(
            y_test,
            y_proba,
            plot_chance_level=is_last_plot,  # display on top
            name=name,
            color=palette[name],
            ax=ax,
        )

        ax.legend()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_pr_curve(results):
    names = list(results)
    palette = _get_palette(names)

    fig, ax = plt.subplots()
    for idx, (name, named_results) in enumerate(results.items()):
        y_proba = named_results["y_proba"][:, 1]
        y_test = named_results["y_test"]
        model = named_results["model"]

        is_last_plot = idx == len(results) - 1

        ax.plot(
            named_results["recall"],
            named_results["precision"],
            marker="o",
            color=palette[name],
            markersize=10,
            label=f"Cut-off point for {named_results['threshold']:.3f}",
        )
        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_proba,
            plot_chance_level=is_last_plot,  # display on top
            name=name,
            color=palette[name],
            ax=ax,
        )
        ax.legend()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(results):
    names = list(results)
    palette = _get_palette(names)

    fig, ax = plt.subplots()
    for name, named_results in results.items():
        y_proba = named_results["y_proba"][:, 1]
        y_test = named_results["y_test"]

        CalibrationDisplay.from_predictions(
            y_test, y_proba, n_bins=10, name=name, ax=ax
        )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_hist_proba(results):
    names = list(results)
    palette = _get_palette(names)

    fig, axes = plt.subplots(figsize=(4, 12), nrows=len(results))
    axes = axes.flatten()
    for ax, (name, named_results) in zip(axes, results.items()):
        y_proba = named_results["y_proba"][:, 1]
        sns.histplot(
            y_proba,
            color=palette[name],
            bins=10,
            ax=ax,
        )
        ax.bar_label(ax.containers[0])
        ax.set_title(name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim([0, 1.1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_permutation_importance(model, X, y, random_state=None, figsize=None):
    perm_result = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        random_state=random_state,
        n_jobs=2,
    )
    mask_nonzero = perm_result.importances_mean > 0
    perm_sorted_idx = perm_result.importances_mean[mask_nonzero].argsort()

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        perm_result.importances[mask_nonzero][perm_sorted_idx].T,
        labels=X.columns[mask_nonzero][perm_sorted_idx],
        vert=False,
    )
    ax.axvline(x=0, color="k", linestyle="--")
    plt.tight_layout()


def plot_confusion_matrix(results, model_names):
    fig, axes = plt.subplots(nrows=1, ncols=len(model_names))

    for model_name, ax in zip(model_names, axes):
        model_result = results[model_name]
        threshold = model_result["model"].best_threshold_
        y_pred = (model_result["y_proba"][:, 1] > threshold).astype("int32")
        y_test = model_result["y_test"]
        ConfusionMatrixDisplay.from_predictions(
            y_pred=y_pred, y_true=y_test, ax=ax, colorbar=False
        )
        ax.set_title(model_name)
    plt.tight_layout()
