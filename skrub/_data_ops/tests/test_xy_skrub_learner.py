import pickle

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import skrub


def test_xy_skrub_learner():
    X, y = make_classification(n_samples=10)

    # regular transformer

    transformer = (
        skrub.var("X")
        .skb.apply(PCA(n_components=2))
        .skb.make_learner()
        .with_sklearn_interface()
    )
    assert not transformer.__sklearn_is_fitted__()
    assert transformer.fit_transform(X).shape == (10, 2)
    assert transformer.__sklearn_is_fitted__()
    assert not hasattr(transformer, "predict")

    # check pickling

    assert pickle.loads(pickle.dumps(transformer)).fit_transform(X).shape == (10, 2)

    # supervised transformer + node id

    X_node = skrub.var("X")
    transformer = (
        X_node.skb.apply(SelectKBest(k=3), y=skrub.var("y"))
        .skb.make_learner()
        .with_sklearn_interface(X_node.skb.id)
    )
    assert transformer.fit_transform(X, y).shape == (10, 3)

    # predictor + different variable names

    predictor = (
        skrub.var("a")
        .skb.apply(PCA(n_components=2))
        .skb.apply(DummyClassifier(), y=skrub.var("b"))
        .skb.make_learner()
        .with_sklearn_interface("a", "b")
    )
    predictor.fit(X, y)
    assert predictor.predict(X).shape == y.shape


def test_bad_X_y_names():
    X = skrub.var("a")
    y = skrub.var("b")
    learner = X.skb.apply(DummyClassifier(), y=y).skb.make_learner()

    with pytest.raises(TypeError, match="Expected X to be a string"):
        learner.with_sklearn_interface(X=np.ones((3, 2)))
    with pytest.raises(TypeError, match="Expected y to be a string"):
        learner.with_sklearn_interface(y=np.ones(3))

    with pytest.raises(ValueError, match="X='z'.*a variable or node named 'z'"):
        learner.with_sklearn_interface(X="z", y="b")
    with pytest.raises(ValueError, match="y='z'.*a variable or node named 'z'"):
        learner.with_sklearn_interface(X="a", y="z")

    with pytest.raises(ValueError, match="X=0.*a variable or node with ID 0"):
        learner.with_sklearn_interface(X=0, y=y.skb.id)
    with pytest.raises(ValueError, match="y=0.*a variable or node with ID 0"):
        learner.with_sklearn_interface(X=X.skb.id, y=0)

    # when y is not needed, no check for y

    learner = X.skb.apply_func(lambda x: x).skb.make_learner()
    learner.with_sklearn_interface(X="a", y=None)
    learner.with_sklearn_interface(X="a", y="y")
    learner.with_sklearn_interface(X="a", y="z")
    learner.with_sklearn_interface(X="a", y=0)


def test_xy_search():
    # wrapping the output of make_*_search in a xylearner
    X, y = make_classification(n_samples=10)
    predictor = (
        skrub.X()
        .skb.apply(PCA(n_components=skrub.choose_from([2, 3], name="n_components")))
        .skb.apply(DummyClassifier(), y=skrub.y())
        .skb.make_grid_search()
        .with_sklearn_interface()
    )
    predictor.fit(X, y)
    assert predictor.results_.shape[0] == 2
    assert isinstance(predictor.best_learner_, skrub.XySkrubLearner)


def test_search_xy_learner():
    # using a xyskrublearner in a scikit-learn searchcv
    X, y = make_classification(n_samples=10)
    pca = PCA(n_components=skrub.choose_from([2, 3], name="n_components"))
    selectk = SelectKBest(k=skrub.choose_from([4, 5], name="k"))
    dim_reduction = skrub.choose_from(
        {"pca": pca, "selectk": selectk}, name="dim_reduction"
    )
    transformer = (
        skrub.var("X")
        .skb.apply(dim_reduction, y=skrub.var("y"))
        .skb.make_learner()
        .with_sklearn_interface()
    )
    pipe = Pipeline([("transformer", transformer), ("predictor", LogisticRegression())])
    outer_grid = {"predictor__C": [0.1, 1.0]}
    param_grid = transformer.get_param_grid("transformer__", outer_grid)
    assert param_grid == [
        {
            "predictor__C": [0.1, 1.0],
            "transformer__2:dim_reduction": ["0:pca"],
            "transformer__0:n_components": ["0:2", "1:3"],
        },
        {
            "predictor__C": [0.1, 1.0],
            "transformer__2:dim_reduction": ["1:selectk"],
            "transformer__1:k": ["0:4", "1:5"],
        },
    ]
    search = GridSearchCV(pipe, param_grid).fit(X, y)
    assert len(search.cv_results_["mean_test_score"]) == 8

    # check other param grid options

    assert transformer.get_param_grid("transformer__", [outer_grid]) == param_grid
    assert transformer.get_param_grid() == [
        {"2:dim_reduction": ["0:pca"], "0:n_components": ["0:2", "1:3"]},
        {"2:dim_reduction": ["1:selectk"], "1:k": ["0:4", "1:5"]},
    ]


def test_get_set_enumerated_param():
    X, y = make_classification(n_samples=10)
    pca = PCA(n_components=skrub.choose_from([2, 3], name="n_components"))
    transformer = (
        skrub.var("X").skb.apply(pca).skb.make_learner().with_sklearn_interface()
    )
    assert transformer.fit_transform(X).shape[1] == 2
    transformer.set_params(learner__data_op__0=1)
    assert transformer.fit_transform(X).shape[1] == 3
    grid = transformer.get_param_grid()
    assert grid == [{"0:n_components": ["0:2", "1:3"]}]
    transformer.set_params(**{"0:n_components": "0:2"})
    assert transformer.fit_transform(X).shape[1] == 2
    assert transformer.get_named_params() == {"n_components": 0}
    transformer.set_named_params(n_components=1)
    assert transformer.fit_transform(X).shape[1] == 3


def test_get_set_numeric_param():
    X, y = make_classification(n_samples=10)
    n_components = skrub.choose_int(2, 8, default=4, name="n_components")
    pca = PCA(n_components=n_components)
    transformer = (
        skrub.var("X").skb.apply(pca).skb.make_learner().with_sklearn_interface()
    )
    assert transformer.fit_transform(X).shape[1] == 4
    transformer.set_params(learner__data_op__0=3)
    assert transformer.fit_transform(X).shape[1] == 3
    grid = transformer.get_param_grid()
    assert (
        str(grid)
        == "[{'0:n_components': choose_int(2, 8, name='n_components', default=4)}]"
    )
    transformer.set_params(**{"0:n_components": 5})
    assert transformer.fit_transform(X).shape[1] == 5
    assert transformer.get_named_params() == {"n_components": 5}
    transformer.set_named_params(n_components=7)
    assert transformer.fit_transform(X).shape[1] == 7


def test_learner_attribute_error():
    # if there is no learner attribute (could happen during unpickling)
    # .learner does not try to look it up in the learner's attribute in
    # __getattr__ and cause infinite recursion.
    xylearner = skrub.var("X").skb.make_learner().with_sklearn_interface()
    del xylearner.learner
    with pytest.raises(AttributeError):
        xylearner.learner
