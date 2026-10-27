"""
IMPORTANT NOTE:

huggingface_hub when downloading a model has a blanket try/except that catches
everything and turns it into an OSError saying the model failed to download.
This masks the real issue.

Our pytest config turns DeprecationWarning and FutureWarning into exceptions,
which then get shown as OSError and are hard to debug.
If you see tests in this module failing with ModelNotFound errors, try
disabling the filterwarnings in pyproject.toml to see if any deprecation
warnings are being raised.
"""

import gc
import pickle
import sys
import time
import weakref

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import clone

import skrub._dataframe as sbd
from skrub import ApplyToCols, TableVectorizer, _llm_encoder
from skrub._llm_encoder import LLMEncoder, ModelNotFound, TextEncoder
from skrub._single_column_transformer import RejectColumn


@pytest.fixture(autouse=True)
def shared_models(monkeypatch):
    """Isolate the models shared by all encoders in each test."""
    models = weakref.WeakValueDictionary()
    monkeypatch.setattr(_llm_encoder, "_MODELS", models)
    return models


class FakeModel:
    def encode(self, X, **kwargs):
        return np.asarray(
            [[len(x), x.count("a"), x.count("e")] for x in X], dtype="float32"
        )


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the sentence-transformers model loading by a cheap fake model,
    and return the list of weak references to the loaded models."""
    loaded = []

    def load(self):
        # make concurrent loads more likely, in case the lock does not work
        time.sleep(0.1)
        model = FakeModel()
        loaded.append(weakref.ref(model))
        return model

    monkeypatch.setattr(LLMEncoder, "_load_estimator", load)
    return loaded


@pytest.fixture
def encoder():
    """Create a fixture such that we set the common parameters of the encoder.

    Here, we have two constraints:

    - We want a really small model to not blow up the memory of the GitHub Actions
      workers.
    - We want to force the use of the CPU. GitHub Actions workers on macOS ARM64 will
      detect the MPS backend, but due to limitations, no memory can be allocated.
      See https://github.com/actions/runner-images/issues/9918 for more details.
    """
    pytest.importorskip("sentence_transformers")
    return LLMEncoder(
        model_name="sentence-transformers/paraphrase-albert-small-v2",
        device="cpu",
    )


def test_missing_import_error(monkeypatch):
    """Test that a clear error is raised when sentence_transformers is missing.

    We mock the missing dependency by hiding it from sys.modules, then verify
    that LLMEncoder.fit() raises an ImportError with a helpful message.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    st = LLMEncoder()  # Direct creation to avoid importorskip
    x = pd.Series(["oh no"])

    err_msg = (
        "Missing optional dependency 'sentence_transformers'.*"
        "LLMEncoder requires sentence-transformers.*"
        "install\\.html#deep-learning-dependencies"
    )
    with pytest.raises(ImportError, match=err_msg):
        st.fit(x)


def test_llm_encoder(df_module, encoder):
    X = df_module.make_column("", ["hello sir", "hola que tal"])
    encoder = clone(encoder).set_params(n_components=2)
    X_out = encoder.fit_transform(X)
    assert sbd.shape(X_out) == (2, 2)

    X_out_2 = encoder.fit_transform(X)
    df_module.assert_frame_equal(X_out, X_out_2)


@pytest.mark.parametrize("X", [["hello"], "hello"])
def test_not_a_series(X, encoder):
    with pytest.raises(ValueError):
        clone(encoder).fit(X)


def test_not_a_series_with_string(df_module, encoder):
    X = df_module.make_column("", [1, 2, 3])
    with pytest.raises(RejectColumn):
        clone(encoder).fit(X)


def test_missing_value(df_module, encoder):
    X = df_module.make_column("", [None, None, "hey"])
    encoder = clone(encoder).set_params(n_components=None)
    X_out = encoder.fit_transform(X)

    assert sbd.shape(X_out) == (3, 768)
    X_out = X_out.to_numpy()
    assert_array_equal(X_out[0, :], X_out[1, :])


def test_n_components(df_module, encoder):
    X = df_module.make_column("", ["hello sir", "hola que tal"])
    encoder_all = clone(encoder).set_params(n_components=None).fit(X)
    for meth in ("fit_transform", "transform"):
        X_out = getattr(encoder_all, meth)(X)
        assert sbd.shape(X_out)[1] == 768
        assert encoder_all.n_components_ == 768

    encoder_2 = clone(encoder).set_params(n_components=2).fit(X)
    for meth in ("fit_transform", "transform"):
        X_out = getattr(encoder_2, meth)(X)
        assert sbd.shape(X_out)[1] == 2
        assert encoder_2.n_components_ == 2

    encoder_30 = clone(encoder).set_params(n_components=30)
    with pytest.warns(UserWarning, match="The embeddings will be truncated"):
        for meth in ("fit_transform", "transform"):
            X_out = getattr(encoder_30, meth)(X)
    assert not hasattr(encoder_30, "pca_")
    assert sbd.shape(X_out)[1] == 30
    assert encoder_30.n_components_ == 30


def test_wrong_parameters(encoder):
    with pytest.raises(ValueError, match="Got n_components='yes'"):
        clone(encoder).set_params(n_components="yes")._check_params()

    with pytest.raises(ValueError, match="Got batch_size=-10"):
        clone(encoder).set_params(batch_size=-10)._check_params()

    with pytest.raises(ValueError, match="Got model_name=1"):
        clone(encoder).set_params(model_name=1)._check_params()

    with pytest.raises(ValueError, match="Got cache_folder=1"):
        clone(encoder).set_params(cache_folder=1)._check_params()

    with pytest.raises(ValueError, match="Got model_name=1"):
        clone(encoder).set_params(model_name=1)._check_params()

    with pytest.raises(ValueError, match="Got verbose=-1"):
        clone(encoder).set_params(verbose=-1)._check_params()

    with pytest.raises(ValueError, match="Got verbose='yes'"):
        clone(encoder).set_params(verbose="yes")._check_params()


def test_verbose_default():
    """The default verbose level is 0 (silent), consistent with the rest of
    skrub (e.g. GapEncoder, TableReport)."""
    assert LLMEncoder().verbose == 0


@pytest.mark.parametrize(
    "verbose, expected_show_progress_bar",
    [(0, False), (1, True), (2, True), (False, False), (True, True)],
)
def test_verbose_controls_progress_bar(
    df_module, verbose, expected_show_progress_bar, monkeypatch
):
    """``verbose`` (int or bool) is forwarded to sentence-transformers as the
    boolean ``show_progress_bar`` flag."""
    calls = []

    class FakeEstimator:
        def encode(self, X, **kwargs):
            calls.append(kwargs["show_progress_bar"])
            return np.zeros((len(X), 3))

    model = FakeEstimator()
    monkeypatch.setattr(LLMEncoder, "_load_estimator", lambda self: model)
    X = df_module.make_column("", ["hello", "hola"])
    encoder = LLMEncoder(n_components=None, verbose=verbose)
    encoder.fit_transform(X)

    assert calls == [expected_show_progress_bar]


def test_wrong_model_name(encoder):
    x = pd.Series(["Good evening Dave"])
    with pytest.raises(ModelNotFound):
        clone(encoder).set_params(model_name="HAL-9000").fit(x)


def test_transform_equal_fit_transform(df_module, encoder):
    x = df_module.make_column("", ["hello again"])
    encoder = clone(encoder).set_params(n_components=None)
    X_out = encoder.fit_transform(x)
    X_out_2 = encoder.transform(x)
    df_module.assert_frame_equal(X_out, X_out_2)


def test_transform_error_on_float_data(df_module, encoder):
    """Check that we raise an error when data without any string is passed at
    transform."""
    x = df_module.make_column("", [1.0, 2.5, 3.7])

    encoder = clone(encoder).set_params(n_components=None)
    encoder.fit(df_module.make_column("", ["hello", "world"]))

    with pytest.raises(ValueError, match="does not contain strings"):
        encoder.transform(x)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_encoder_in_table_vectorizer(df_module, encoder, n_jobs):
    X = df_module.make_dataframe({"msg": ["hey there", "hello"]})
    X_out = TableVectorizer(low_cardinality=encoder, n_jobs=n_jobs).fit_transform(X)
    assert sbd.shape(X_out) == (2, 30)


def test_token_env_variable(df_module, encoder):
    token_env_variable = "my_imaginary_token"
    X = df_module.make_column("", ["hello", "hola"])
    encoder = clone(encoder).set_params(token_env_variable=token_env_variable).fit(X)
    assert encoder.token_env_variable == token_env_variable


@pytest.mark.parametrize("store_weights_in_pickle", [True, False])
def test_store_weights_in_pickle(df_module, encoder, store_weights_in_pickle):
    X = df_module.make_column("", ["hello", "hola", "guttentag"])
    encoder = (
        clone(encoder)
        .set_params(
            store_weights_in_pickle=store_weights_in_pickle,
            n_components=2,
        )
        .fit(X)
    )
    assert hasattr(encoder, "_estimator")
    assert ("_estimator" in encoder.__getstate__()) is store_weights_in_pickle

    obj = pickle.dumps(encoder)
    encoder_unpickled = pickle.loads(obj)
    # the model is already loaded in this process, so it is shared
    assert encoder_unpickled._estimator is encoder._estimator
    df_module.assert_frame_equal(encoder_unpickled.transform(X), encoder.transform(X))


def test_unpickle_stored_weights_are_shared(df_module, fake_model, shared_models):
    X = df_module.make_column("", ["hello", "hola", "guttentag"])
    encoder = LLMEncoder(n_components=None, store_weights_in_pickle=True).fit(X)
    obj = pickle.dumps(encoder)
    del encoder
    gc.collect()
    assert len(shared_models) == 0

    encoder_unpickled = pickle.loads(obj)
    # the pickled weights are shared rather than loading the model again
    assert len(fake_model) == 1
    assert list(shared_models.values()) == [encoder_unpickled._estimator]
    assert LLMEncoder(n_components=None).fit(X)._estimator is (
        encoder_unpickled._estimator
    )
    assert len(fake_model) == 1


def test_model_released_when_unused(df_module, fake_model, shared_models):
    X = df_module.make_column("", ["hello", "hola"])
    encoder_1 = LLMEncoder(n_components=None).fit(X)
    encoder_2 = LLMEncoder(n_components=None).fit(X)
    assert encoder_1._estimator is encoder_2._estimator
    assert len(fake_model) == 1

    # the model is kept while at least one encoder uses it
    del encoder_1
    gc.collect()
    assert fake_model[0]() is encoder_2._estimator

    del encoder_2
    gc.collect()
    assert fake_model[0]() is None
    assert len(shared_models) == 0

    # it is loaded again when needed
    LLMEncoder(n_components=None).fit(X)
    assert len(fake_model) == 2


def test_model_loaded_once_for_several_columns(df_module, fake_model):
    X = df_module.make_dataframe(
        {"a": ["hello", "hola"], "b": ["hallo", "ciao"], "c": ["hej", "salut"]}
    )
    for transformer in [
        ApplyToCols(LLMEncoder(n_components=None)),
        TableVectorizer(high_cardinality=LLMEncoder(n_components=None)),
    ]:
        transformer.fit(X)
    assert len(fake_model) == 1


def test_model_loaded_once_with_threads(df_module, fake_model):
    X = df_module.make_dataframe(
        {f"col_{i}": ["hello", f"hola {i}", "guten Tag"] for i in range(8)}
    )
    with joblib.parallel_backend("threading"):
        ApplyToCols(LLMEncoder(n_components=None), n_jobs=8).fit(X)
    assert len(fake_model) == 1


def test_models_shared_by_name_and_device(df_module, fake_model):
    X = df_module.make_column("", ["hello", "hola"])
    encoders = [
        LLMEncoder(model_name="a", n_components=None),
        LLMEncoder(model_name="a", n_components=None),
        LLMEncoder(model_name="b", n_components=None),
        LLMEncoder(model_name="a", device="cpu", n_components=None),
    ]
    models = [enc.fit(X)._estimator for enc in encoders]
    assert len(fake_model) == 3
    assert models[0] is models[1]
    assert len({id(m) for m in models}) == 3


def test_categorical_features(df_module, encoder):
    cat_col = sbd.to_categorical(
        df_module.make_column("cat", ["A", "B", "A", "C", "B", "D"])
    )
    data = {
        "categorical": cat_col,
        "numeric": [1, 2, 3, 4, 5, 6],
    }
    df = df_module.make_dataframe(data)

    with pytest.raises(RejectColumn):
        encoder.fit(df["numeric"])

    out = encoder.fit_transform(df["categorical"])
    assert len(sbd.column_names(out)) == 30

    out = encoder.fit(df["categorical"][:4]).transform(df["categorical"][4:])
    assert len(sbd.column_names(out)) == 30


def test_deprecated_text_encoder_warning(df_module):
    # We need to define a new encoder with the proper class rather than reusing
    # the fixture (which is a LLMEncoder)
    with pytest.warns(FutureWarning, match="TextEncoder is deprecated.*"):
        TextEncoder(
            model_name="sentence-transformers/paraphrase-albert-small-v2",
            device="cpu",
        )
