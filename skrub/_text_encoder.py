import functools
import numbers
import os
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


class TextEncoder(SingleColumnTransformer, TransformerMixin):
    """Encode string features by applying a pretrained language model \
        downloaded from the HuggingFace Hub.

    This is a thin wrapper around :class:`~sentence_transformers.SentenceTransformer`
    that follows the scikit-learn API, making it usable within a scikit-learn pipeline.

    .. warning::

        To use this class, you need to install the optional ``transformers``
        dependencies for skrub. See the "deep learning dependencies" section
        in the :ref:`installation_instructions` guide for more details.

    This class uses a pre-trained model, so calling ``fit`` or ``fit_transform``
    will not train or fine-tune the model. Instead, the model is loaded from disk,
    and a PCA is fitted to reduce the dimension of the language model's output,
    if ``n_components`` is not None.

    When PCA is disabled, this class is essentially stateless, with loading the
    pre-trained model from disk being the only difference between ``fit_transform``
    and ``transform``.

    Be aware that parallelizing this class (e.g., using
    :class:`~skrub.TableVectorizer` with ``n_jobs`` > 1) may be computationally
    expensive. This is because a copy of the pre-trained model is loaded into memory
    for each thread. Therefore, we recommend you to let the default n_jobs=None
    (or set to 1) of the TableVectorizer and let pytorch handle parallelism.

    If memory usage is a concern, check the characteristics of your selected model.

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

        The default model is a shrunk version of e5-v2, which has shown good
        performance in the benchmark of [1]_.

    n_components : int or None, default=30,
        The number of embedding dimensions. As the number of dimensions is different
        across embedding models, this class uses a :class:`~sklearn.decomposition.PCA`
        to set the number of embedding to ``n_components`` during ``transform``.
        Set ``n_components=None`` to skip the PCA dimension reduction mechanism.

        See [1]_ for more details on the choice of the PCA and default
        ``n_components``.

    device : str, default=None
        Device (e.g. "cpu", "cuda", "mps") that should be used for computation.
        If None, checks if a GPU can be used.
        Note that macOS ARM64 users can enable the GPU on their local machine
        by setting ``device="mps"``.

    batch_size : int, default=32
        The batch size to use during ``transform``.

    token_env_variable : str, default=None
        The name of the environment variable which stores your HuggingFace
        authentication token to download private models.
        Note that we only store the name of the variable but not the token itself.

    cache_folder : str, default=None
        Path to store models. By default ``~/skrub_data``.
        See :func:`skrub.datasets._utils.get_data_dir`.
        Note that when unpickling ``TextEncoder`` on another machine,
        the ``cache_folder`` path needs to be accessible to store the downloaded model.

    store_weights_in_pickle : bool, default=False
        Whether or not to keep the loaded sentence-transformers model
        in the ``TextEncoder`` when pickling.

        - When set to False, the ``_estimator`` property is removed from
          the object to pickle, which significantly reduces the size of
          the serialized object. Note that when the serialized object is
          unpickled on another machine, the ``TextEncoder`` will try to download
          the sentence-transformer model again from HuggingFace Hub.
          This process could fail if, for example, the machine doesn't have
          internet access. Additionally, if you use weights stored on disk
          that are *not* on the HuggingFace Hub (by passing a path to
          ``model_name``), these weights will not be pickled either.
          Therefore you would need to copy them to the machine where you
          unpickle the ``TextEncoder``.
        - When set to True, the ``_estimator`` property is included in
          the serialized object. Users deploying fine-tuned models stored on
          disk are recommended to use this option. Note that the machine
          where the ``TextEncoder`` is unpickled must have the same device than
          the machine where it was pickled.

    random_state : int, RandomState instance or None, default=None
        Used when the PCA dimension reduction mechanism is used, for reproducible
        results across multiple function calls.

    verbose : bool, default=True
        Verbose level, controls whether to show a progress bar or not during
        ``transform``.

    Attributes
    ----------
    input_name_ : str
        The name of the fitted column.

    pca_ : sklearn.decomposition.PCA
        A fitted PCA to reduce the embedding dimensionality (either PCA or truncation,
        see the ``n_components`` parameter).

    n_components_ : int
        The number of dimensions of the embeddings after dimensionality
        reduction.

    See Also
    --------
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    GapEncoder :
        Encode string columns by constructing latent topics.
    StringEncoder
        Fast n-gram encoding of string columns.
    SimilarityEncoder :
        Encode string columns as a numeric array with n-gram string similarity.

    References
    ----------
    .. [1]  L. Grinsztajn, M. Kim, E. Oyallon, G. Varoquaux
            "Vectorizing string entries for data processing on tables: when are larger
            language models better?", 2023.
            https://hal.science/hal-04345931

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import TextEncoder

    Let's encode video comments using only 2 embedding dimensions:

    >>> enc = TextEncoder(model_name='intfloat/e5-small-v2', n_components=2)
    >>> X = pd.Series([
    ...   "The professor snatched a good interview out of the jaws of these questions.",
    ...   "Bookmarking this to watch later.",
    ...   "When you don't know the lyrics of the song except the chorus",
    ... ], name='video comments')

    Fitting does not train the underlying pre-trained deep-learning model,
    but ensure various checks and enable dimension reduction.

    >>> enc.fit_transform(X) # doctest: +SKIP
       video comments_1  video comments_2
    0          0.411395          0.096504
    1         -0.105210         -0.344567
    2         -0.306184          0.248063
    """

    def __init__(
        self,
        model_name="intfloat/e5-small-v2",
        n_components=30,
        device=None,
        batch_size=32,
        token_env_variable=None,
        cache_folder=None,
        store_weights_in_pickle=False,
        random_state=None,
        verbose=False,
    ):
        self.model_name = model_name
        self.n_components = n_components
        self.device = device
        self.batch_size = batch_size
        self.token_env_variable = token_env_variable
        self.cache_folder = cache_folder
        self.store_weights_in_pickle = store_weights_in_pickle
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, column, y=None):
        """Fit the TextEncoder from ``column``.

        In practice, it loads the pre-trained model from disk and returns
        the embeddings of the column.

        Parameters
        ----------
        column : pandas or polars Series of shape (n_samples,)
            The string column to compute embeddings from.

        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        X_out : pandas or polars DataFrame of shape (n_samples, n_components)
            The embedding representation of the input.
        """
        del y
        if not sbd.is_string(column):
            raise RejectColumn(f"Column {sbd.name(column)!r} does not contain strings.")

        self._check_params()

        self.input_name_ = sbd.name(column)

        X_out = self._vectorize(column)

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
                # self.n_components can be greater than the number
                # of dimensions of X_out.
                # Therefore, self.n_components_ below stores the resulting
                # number of dimensions of X_out.
                X_out = X_out[:, : self.n_components]

        self.n_components_ = X_out.shape[1]

        cols = self.get_feature_names_out()
        X_out = sbd.make_dataframe_like(column, dict(zip(cols, X_out.T)))
        X_out = sbd.copy_index(column, X_out)

        return X_out

    def transform(self, column):
        """Transform ``column`` using the TextEncoder.

        This method uses the embedding model loaded in memory during ``fit``
        or ``fit_transform``.

        Parameters
        ----------
        column : pandas or polars Series of shape (n_samples,)
            The string column to compute embeddings from.

        Returns
        -------
        X_out : pandas or polars DataFrame of shape (n_samples, n_components)
            The embedding representation of the input.
        """
        check_is_fitted(self, "_estimator")

        if not sbd.is_string(column):
            raise ValueError(f"Column {sbd.name(column)!r} does not contain strings.")

        X_out = self._vectorize(column)

        if hasattr(self, "pca_"):
            X_out = self.pca_.transform(X_out)
        elif self.n_components is not None:
            X_out = X_out[:, : self.n_components]

        cols = self.get_feature_names_out()
        X_out = sbd.make_dataframe_like(column, dict(zip(cols, X_out.T)))
        X_out = sbd.copy_index(column, X_out)

        return X_out

    def _vectorize(self, column):
        is_null = sbd.to_numpy(sbd.is_null(column))
        column = sbd.to_numpy(column)
        unique_x, indices_x = unique_strings(column, is_null)

        # sentence-transformers deals with converting a torch tensor
        # to a numpy array, on CPU.
        return self._estimator.encode(
            unique_x,
            normalize_embeddings=False,
            batch_size=self.batch_size,
            show_progress_bar=self.verbose,
        )[indices_x]

    @functools.cached_property
    def _estimator(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*IProgress not found.*",
            )
            st = import_optional_dependency("sentence_transformers")

        self._cache_folder = get_data_dir(
            name=self.model_name, data_home=self.cache_folder
        )

        if self.token_env_variable is not None:
            token = os.getenv(self.token_env_variable)
        else:
            token = None

        try:
            estimator = st.SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self._cache_folder,
                token=token,
            )
        except OSError as e:
            raise ModelNotFound(
                f"{self.model_name} is not a local folder and is not a valid "
                "model identifier listed on 'https://huggingface.co/models'.\n "
                "If this is a private repository, make sure to pass a token having "
                "permission to this repo by setting this token as an environment "
                "variable, and passing this variable to the TextEncoder as "
                "`token_env_variable=<your_token_env_variable>`"
            ) from e
        return estimator

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

    def __getstate__(self):
        state = self.__dict__.copy()
        # Always dump self._cache_folder because it is overwritten when the model
        # is loaded, and it shows an absolute path on the user machine.
        # However, we have to include self.cache_folder in the serialized object
        # because that is a parameter provided by the user.
        remove_props = ["_cache_folder"]
        if not self.store_weights_in_pickle:
            remove_props.append("_estimator")

        for prop in remove_props:
            if prop in state:
                del state[prop]

        return state
