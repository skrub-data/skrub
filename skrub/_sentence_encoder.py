import numbers
import warnings
from pathlib import Path

from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer
from ._utils import import_optional_dependency, unique_strings
from .datasets._utils import get_data_dir


class ModelNotFound(ValueError):
    pass


class SentenceEncoder(SingleColumnTransformer, TransformerMixin):
    """Encode string features by applying an embedding model downloaded \
        from the HuggingFace Hub.

    This is a thin wrapper around :class:`~sentence_transformers.SentenceTransformer`
    that follows the scikit-learn API, making it usable within a scikit-learn pipeline.

    .. warning::

        To use this class, you need to install the optional ``transformers``
        dependencies for skrub. For example, you can use pip:

        .. code::

           pip install skrub[transformers] -U

        This will install the ``sentence-transformers``, ``transformers``,
        and ``torch`` libraries on your machine. Be aware that this might lead
        to dependency conflicts.


    This class uses a pre-trained model, so calling ``fit`` or ``fit_transform``
    will not train or fine-tune the model. Instead, the model is loaded from disk,
    and a PCA is fitted if ``n_components`` is not None.

    When PCA is disabled, this class is essentially stateless, with loading the
    pre-trained model from disk being the only difference between ``fit_transform``
    and ``transform``.

    Be aware that parallelizing this class (e.g., using
    :class:~skrub.TableVectorizer with ``n_jobs`` > 1) may be computationally expensive.
    This is because a copy of the pre-trained model is loaded into memory
    for each thread. If memory usage is a concern, check the characteristics of
    your selected model.

    Parameters
    ----------
    model_name : str, default="intfloat/e5-small-v2"

        - If a filepath on disk is passed, this class loads the model from that path.
        - Otherwise, it first tries to download a pre-trained
          :class:`~sentence_transformers.SentenceTransformer` model.
          If that fails, tries to construct a model from Huggingface models repository
          with that name.

        The following models have a good performance/memory usage tradeoff:

        - ``intfloat/e5-small-v2``
        - ``all-MiniLM-L6-v2``
        - ``all-mpnet-base-v2``

        You can find more options on the `sentence-transformers documentation
        <https://www.sbert.net/docs/pretrained_models.html#model-overview>`_.

        The default model is a shrinked version of e5-v2, which has shown good
        performance in the benchmark of [1]_.

    n_components : int or None, default=30,
        The number of embedding dimensions. As the number of dimensions is different
        accross embedding models, this class uses a :class:`~sklearn.decomposition.PCA`
        to set the number of embedding to ``n_components`` during ``transform``.
        Set ``n_components=None`` to skip the PCA dimension reduction mechanism.

        See [1]_ for more details on the choice of the PCA and default ``n_components``.

    device : str, default=None
        Device (e.g. "cpu", "cuda", "mps") that should be used for computation. If None,
        checks if a GPU can be used.
        Note that macOS ARM64 users can enable the GPU on their local machine
        by setting ``device="mps"``.

    batch_size : int, default=32
        The batch size to use during ``transform``.

    use_auth_token : bool or str, default=None
        HuggingFace authentication token to download private models.

    cache_folder : str, default=None
        Path to store models. By default ``~/skrub_data``.
        See :func:`~skrub.datasets._utils.get_data_dir`.

    random_state : int, RandomState instance or None, default=None
        Used when the PCA dimension reduction mechanism is used, for reproducible
        results across multiple function calls.

    verbose : bool, default=True
        Verbose level, controls whether to show a progress bar or not during
        ``transform``.

    References
    ----------
    .. [1]  L. Grinsztajn, M. Kim, E. Oyallon, G. Varoquaux
            "Vectorizing string entries for data processing on tables: when are larger
            language models better?", 2023.
            https://hal.science/hal-04345931
    """

    def __init__(
        self,
        model_name="intfloat/e5-small-v2",
        n_components=30,
        device=None,
        batch_size=32,
        norm="l2",
        use_auth_token=None,
        cache_folder=None,
        random_state=None,
        verbose=False,
    ):
        self.model_name = model_name
        self.n_components = n_components
        self.device = device
        self.batch_size = batch_size
        self.norm = norm
        self.use_auth_token = use_auth_token
        self.cache_folder = cache_folder
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, X, y=None):
        """Fit the SentenceEncoder from ``X``.

        In practice, it loads the pre-trained model from disk and returns
        the embeddings of X.

        Parameters
        ----------
        X : pandas or polars Series of shape (n_samples,)
            The string column to compute embeddings from.

        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        X_out : pandas or polars DataFrame of shape (n_samples, n_components)
            The embedding representation of the input.
        """
        st = import_optional_dependency("sentence_transformers")

        if not sbd.is_string(X):
            raise RejectColumn(f"Column {sbd.name(X)!r} does not contain strings.")

        self._check_params()

        self.cache_folder_ = get_data_dir(
            name=self.model_name, data_home=self.cache_folder
        )
        try:
            self.estimator_ = st.SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder_,
                use_auth_token=self.use_auth_token,
            )
        except OSError as e:
            raise ModelNotFound(
                f"{self.model_name} is not a local folder and is not a valid "
                "model identifier listed on 'https://huggingface.co/models'.\n "
                "If this is a private repository, make sure to pass a token having "
                "permission to this repo by passing `use_auth_token=<your_token>`"
            ) from e

        self.input_name_ = sbd.name(X)

        X_out = self._vectorize(X)

        if self.n_components is not None:
            if (min_shape := min(X_out.shape)) >= self.n_components:
                self.pca_ = PCA(
                    n_components=self.n_components,
                    copy=False,
                    random_state=self.random_state,
                )
                X_out = self.pca_.fit_transform(X_out)
            else:
                warnings.warn(
                    f"The matrix shape is {(X_out.shape)}, and its minimum is "
                    f"{min_shape}, which is too small to fit a PCA with "
                    f"n_components={self.n_components}. "
                    "The embeddings will be truncated by keeping the first "
                    f"{self.n_components} dimensions instead. "
                    "Set n_components=None to keep all dimensions and remove "
                    "this warning."
                )
                X_out = X_out[:, : self.n_components]

        self.n_components_ = X_out.shape[1]

        cols = self.get_feature_names_out()
        X_out = sbd.make_dataframe_like(X, dict(zip(cols, X_out.T)))
        X_out = sbd.copy_index(X, X_out)

        return X_out

    def transform(self, X):
        """Transform ``X`` using the SentenceEncoder.

        This method uses the embedding model loaded in memory during ``fit``
        or ``fit_transform``.

        Parameters
        ----------
        X : pandas or polars Series of shape (n_samples,)
            The string column to compute embeddings from.

        Returns
        -------
        X_out : pandas or polars DataFrame of shape (n_samples, n_components)
            The embedding representation of the input.
        """
        check_is_fitted(self, "estimator_")

        if not sbd.is_string(X):
            raise RejectColumn(f"Column {sbd.name(X)!r} does not contain strings.")

        X_out = self._vectorize(X)

        if hasattr(self, "pca_"):
            X_out = self.pca_.transform(X_out)
        elif self.n_components != "all":
            X_out = X_out[:, : self.n_components]

        cols = self.get_feature_names_out()
        X_out = sbd.make_dataframe_like(X, dict(zip(cols, X_out.T)))
        X_out = sbd.copy_index(X, X_out)

        return X_out

    def _vectorize(self, X):
        is_null = sbd.to_numpy(sbd.is_null(X))
        X = sbd.to_numpy(X)
        unique_x, indices_x = unique_strings(X, is_null)

        return self.estimator_.encode(
            unique_x,
            normalize_embeddings=False,
            batch_size=self.batch_size,
            show_progress_bar=self.verbose,
        )[indices_x]

    def _check_params(self):
        # XXX: Use sklearn _parameter_constraints instead?
        if self.n_components is not None and not isinstance(
            self.n_components, numbers.Integral
        ):
            raise ValueError(
                f"Got n_components={self.n_components!r} but expected an integer "
                "or None."
            )

        if not (isinstance(self.batch_size, numbers.Integral) and self.batch_size > 0):
            raise ValueError(
                f"Got batch_size={self.batch_size} but expected a positive integer"
            )

        if self.cache_folder is not None and not isinstance(
            self.cache_folder, (str, bytes, Path)
        ):
            raise ValueError(
                f"Got cache_folder={self.cache_folder} but expected a "
                "str, bytes or Path type."
            )

        if not isinstance(self.model_name, (str, Path)):
            raise ValueError(
                f"Got model_name={self.model_name} but expected a str or a Path type."
            )
        return

    def get_feature_names_out(self):
        """Get output feature names for transformation.

        Returns
        -------
        feature_names_out : list of str
            Transformed feature names.
        """
        check_is_fitted(self)
        n_digits = len(str(self.n_components_))
        return [
            f"{self.input_name_}_{str(i).zfill(n_digits)}"
            for i in range(1, self.n_components_ + 1)
        ]
