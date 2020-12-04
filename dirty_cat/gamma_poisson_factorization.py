"""
Online Gamma-Poisson factorization of string arrays.
The principle is as follows:
    1. Given an input string data X, we build its bag-of-n-grams
       representation V (n_samples, vocab_size).
    2. Instead of using the n-grams counts as encodings, we look for low-
       dimensional representations by modeling n-grams counts as as linear
       combinations of topics V = HW, with W (n_topics, vocab_size) the topics
       and H (n_samples, n_topics) the associated activations.
    3. Assuming that n-grams counts follow a Poisson law, we fit H and W to
       maximize the likelihood of the data, with a Gamma prior for the
       activations H to induce sparsity.
    4. In practice, this is equivalent to a non-negative matrix factorization
       with the Kullback-Leibler divergence as loss, and a Gamma prior on H.
       We thus optimize H and W with the multiplicative update method.
"""
import numpy as np
from distutils.version import LooseVersion
from scipy import sparse
import sklearn
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

if LooseVersion(sklearn.__version__) < LooseVersion('0.22'):
    from sklearn.cluster.k_means_ import _k_init
elif LooseVersion(sklearn.__version__) < LooseVersion('0.24'):
    from sklearn.cluster._kmeans import _k_init
else:
    from sklearn.cluster._kmeans import kmeans_plusplus

if LooseVersion(sklearn.__version__) < LooseVersion('0.22'):
    from sklearn.decomposition.nmf import _beta_divergence
else:
    from sklearn.decomposition._nmf import _beta_divergence


class OnlineGammaPoissonFactorization(BaseEstimator, TransformerMixin):
    """
    Online Gamma-Poisson Factorization by minimizing the
    Kullback-Leibler divergence.

    Parameters
    ----------

    n_topics : int, default=10
        Number of topics of the matrix factorization.

    batch_size : int, default=512
        Number of samples per batch.

    gamma_shape_prior : float, default=1.1
        Shape parameter for the Gamma prior distribution.

    gamma_scale_prior : float, default=1.0
        Scale parameter for the Gamma prior distribution.

    rho : float, default=0.95
        Weight parameter for the update of the W matrix.
    
    rescale_rho : boolean, default=False
        If true, use rho ** (batch_size / len(X)) instead of rho to obtain an
        update rate per iteration that is independent of the batch size.

    hashing : boolean, default=False
        If true, HashingVectorizer is used instead of CountVectorizer.

    hashing_n_features : int, default=2**12
        Number of features for the HashingVectorizer. Only relevant if
        hashing=True.

    init : str, default='k-means++'
        Initialization method of the W matrix.
        Options: {'k-means++', 'random', 'k-means'}.
        If init='k-means++', we use the init method of sklearn.cluster.KMeans.
        If init='random', topics are initialized with a Gamma distribution.
        If init='k-means', topics are initialized with a KMeans on the n-grams
        counts. This usually makes convergence faster but is a bit slower.

    tol : float, default=1e-4
        Tolerance for the convergence of the matrix W.

    min_iter : int, default=2
        Minimum number of iterations on the input data.

    max_iter : int, default=5
        Maximum number of iterations on the input data.

    ngram_range : tuple, default=(2, 4)
        The range of ngram length that will be used to build the
        bag-of-n-grams representation of the input data.

    analyzer : str, default='char'.
        Analyzer parameter for the CountVectorizer/HashingVectorizer.
        Options: {‘word’, ‘char’, ‘char_wb’}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option ‘char_wb’ creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.

    add_words : boolean, default=False
        If true, add the words counts to the bag-of-n-grams representation
        of the input data.

    random_state : int or None, default=None
        Pass an int for reproducible output across multiple function calls.

    rescale_W : boolean, default=True
        If true, the weight matrix W is rescaled at each iteration
        to have an l1 norm equal to 1 for each row.
    
    max_iter_e_step : int, default=20
        Maximum number of iterations to adjust the activations h at each step.


    Attributes
    ----------

    References
    ----------
    For a detailed description of the method, see
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ by Cerda, Varoquaux (2019).
    """

    def __init__(self, n_topics=10, batch_size=512, gamma_shape_prior=1.1,
                 gamma_scale_prior=1.0, rho=.95, rescale_rho=False,
                 hashing=False, hashing_n_features=2**12, init='k-means++',
                 tol=1e-4, min_iter=2, max_iter=5, ngram_range=(2, 4),
                 analyzer='char', add_words=False, random_state=None,
                 rescale_W=True, max_iter_e_step=20):

        self.ngram_range = ngram_range
        self.n_topics = n_topics
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.rho = rho
        self.rescale_rho = rescale_rho
        self.batch_size = batch_size
        self.tol = tol
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init = init
        self.analyzer = analyzer
        self.add_words = add_words
        self.random_state = check_random_state(random_state)
        self.rescale_W = rescale_W
        self.max_iter_e_step = max_iter_e_step

        if self.hashing:
            self.ngrams_count = HashingVectorizer(
                 analyzer=self.analyzer, ngram_range=self.ngram_range,
                 n_features=self.hashing_n_features,
                 norm=None, alternate_sign=False)
            if self.add_words:
                self.word_count = HashingVectorizer(
                     analyzer='word',
                     n_features=self.hashing_n_features,
                     norm=None, alternate_sign=False)
        else:
            self.ngrams_count = CountVectorizer(
                 analyzer=self.analyzer, ngram_range=self.ngram_range)
            if self.add_words:
                self.word_count = CountVectorizer()

    def _update_H_dict(self, X, H):
        """
        For each category x in X, update the dictionary self.H_dict with
        the corresponding bag-of-n-grams representation h.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The string data to fit the model on.
        H : array-like, shape (n_samples, n_vocab)
            The corresponding bag-of-n-grams representations.
        """
        for x, h in zip(X, H):
            self.H_dict[x] = h

    def _init_vars(self, X):
        """
        Build the bag-of-n-grams representation H of X and initialize
        the topics W.
        """
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count.fit_transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.fit_transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

        if not self.hashing:
            self.vocabulary = self.ngrams_count.get_feature_names()
            if self.add_words:
                self.vocabulary = np.concatenate(
                    (self.vocabulary, self.word_count.get_feature_names()))

        _, self.n_vocab = unq_V.shape
        self.W_, self.A_, self.B_ = self._init_w(unq_V[lookup], X)
        unq_H = _rescale_h(unq_V, np.ones((len(unq_X), self.n_topics)))
        self.H_dict = dict()
        self._update_H_dict(unq_X, unq_H)
        if self.rescale_rho:
            self.rho_ = self.rho ** (self.batch_size / len(X))
        else:
            self.rho_ = self.rho
        return unq_X, unq_V, lookup

    def _get_H(self, X):
        """
        Return the bag-of-n-grams representation of X.
        """
        H_out = np.empty((len(X), self.n_topics))
        for x, h_out in zip(X, H_out):
            h_out[:] = self.H_dict[x]
        return H_out

    def _init_w(self, V, X):
        """
        Initialize the topics W.
        If self.init='k-means++', we use the init method of
        sklearn.cluster.KMeans.
        If self.init='random', topics are initialized with a Gamma
        distribution.
        If self.init='k-means', topics are initialized with a KMeans on the
        n-grams counts.
        """
        if self.init == 'k-means++':
            if LooseVersion(sklearn.__version__) < LooseVersion('0.24'):
                W = _k_init(
                    V, self.n_topics,
                    x_squared_norms=row_norms(V, squared=True),
                    random_state=self.random_state,
                    n_local_trials=None) + .1
            else:
                W, _ = kmeans_plusplus(
                    V, self.n_topics,
                    x_squared_norms=row_norms(V, squared=True),
                    random_state=self.random_state,
                    n_local_trials=None)
                W += .1
        elif self.init == 'random':
            W = self.random_state.gamma(
                shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
                size=(self.n_topics, self.n_vocab))
        elif self.init == 'k-means':
            prototypes = get_kmeans_prototypes(
                X, self.n_topics, random_state=self.random_state)
            W = self.ngrams_count.transform(prototypes).A + .1
            if self.add_words:
                W2 = self.word_count.transform(prototypes).A + .1
                W = np.hstack((W, W2))
            # if k-means doesn't find the exact number of prototypes
            if W.shape[0] < self.n_topics:
                if LooseVersion(sklearn.__version__) < LooseVersion('0.24'):
                    W2 = _k_init(
                        V, self.n_topics - W.shape[0],
                        x_squared_norms=row_norms(V, squared=True),
                        random_state=self.random_state,
                        n_local_trials=None) + .1
                else:
                    W2, _ = kmeans_plusplus(
                        V, self.n_topics - W.shape[0],
                        x_squared_norms=row_norms(V, squared=True),
                        random_state=self.random_state,
                        n_local_trials=None)
                    W2 += .1
                W = np.concatenate((W, W2), axis=0)
        else:
            raise AttributeError(
                'Initialization method %s does not exist.' % self.init)
        W /= W.sum(axis=1, keepdims=True)
        A = np.ones((self.n_topics, self.n_vocab)) * 1e-10
        B = A.copy()
        return W, A, B

    def fit(self, X, y=None):
        """
        Fit the OnlineGammaPoissonFactorization to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, 1)
            The string data to fit the model on.
        
        Returns
        -------
        self
        """
        X = np.asarray(X)
        assert X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1), f"ERROR:\
        shape {X.shape} of input array is not supported."
        if X.ndim == 2:
            X = X[:, 0]
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "ERROR: Input data is not string."
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
        del X
        unq_H = self._get_H(unq_X)

        for iter in range(self.max_iter):
            for i, (unq_idx, idx) in enumerate(batch_lookup(
              lookup, n=self.batch_size)):
                if i == n_batch-1:
                    W_last = self.W_.copy()
                unq_H[unq_idx] = _multiplicative_update_h(
                    unq_V[unq_idx], self.W_, unq_H[unq_idx],
                    epsilon=1e-3, max_iter=self.max_iter_e_step,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior)
                _multiplicative_update_w(
                    unq_V[idx], self.W_, self.A_, self.B_, unq_H[idx],
                    self.rescale_W, self.rho_)

                if i == n_batch-1:
                    W_change = np.linalg.norm(
                        self.W_ - W_last) / np.linalg.norm(W_last)

            if (W_change < self.tol) and (iter >= self.min_iter - 1):
                break

        self._update_H_dict(unq_X, unq_H)
        return self

    # def get_feature_names(self, n_top=3):
    #     vectorizer = CountVectorizer()
    #     vectorizer.fit(list(self.H_dict.keys()))
    #     vocabulary = np.array(vectorizer.get_feature_names())
    #     encoding = self.transform(np.array(vocabulary).reshape(-1))
    #     encoding = abs(encoding)
    #     encoding = encoding / np.sum(encoding, axis=1, keepdims=True)
    #     n_components = encoding.shape[1]
    #     topic_labels = []
    #     for i in range(n_components):
    #         x = encoding[:, i]
    #         labels = vocabulary[np.argsort(-x)[: n_top]]
    #         topic_labels.append(labels)
    #     topic_labels = [', '.join(label) for label in topic_labels]
    #     return topic_labels

    def score(self, X):
        """
        Returns the Kullback-Leibler divergence.

        Parameters
        ----------
        X : array-like (str), shape (n_samples, )
            The data to encode.

        Returns
        -------
        kl_divergence : float.
            The Kullback-Leibler divergence.
        """

        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        for slice in gen_batches(n=unq_H.shape[0],
                                 batch_size=self.batch_size):
            unq_H[slice] = _multiplicative_update_h(
                unq_V[slice], self.W_, unq_H[slice],
                epsilon=1e-3, max_iter=self.max_iter_e_step,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior)
        kl_divergence = _beta_divergence(
            unq_V[lookup], unq_H[lookup], self.W_,
            'kullback-leibler', square_root=False)
        return kl_divergence

    # def partial_fit(self, X, y=None):
    #     assert X.ndim == 1
    #     if hasattr(self, 'vocabulary'):
    #         unq_X, lookup = np.unique(X, return_inverse=True)
    #         unq_V = self.ngrams_count.transform(unq_X)
    #         if self.add_words:
    #             unq_V2 = self.word_count.transform(unq_X)
    #             unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

    #         unseen_X = np.setdiff1d(unq_X, np.array([*self.H_dict]))
    #         unseen_V = self.ngrams_count.transform(unseen_X)
    #         if self.add_words:
    #             unseen_V2 = self.word_count.transform(unseen_X)
    #             unseen_V = sparse.hstack((unseen_V, unseen_V2), format='csr')

    #         if unseen_V.shape[0] != 0:
    #             unseen_H = _rescale_h(
    #                 unseen_V, np.ones((len(unseen_X), self.n_topics)))
    #             for x, h in zip(unseen_X, unseen_H):
    #                 self.H_dict[x] = h
    #             del unseen_H
    #         del unseen_X, unseen_V
    #     else:
    #         unq_X, unq_V, lookup = self._init_vars(X)
    #         self.rho_ = self.rho

    #     unq_H = self._get_H(unq_X)
    #     unq_H = _multiplicative_update_h(
    #         unq_V, self.W_, unq_H,
    #         epsilon=1e-3, max_iter=self.max_iter_e_step,
    #         rescale_W=self.rescale_W,
    #         gamma_shape_prior=self.gamma_shape_prior,
    #         gamma_scale_prior=self.gamma_scale_prior)
    #     self._update_H_dict(unq_X, unq_H)
    #     _multiplicative_update_w(
    #         unq_V[lookup], self.W_, self.A_, self.B_,
    #         unq_H[lookup], self.rescale_W, self.rho_)
    #     return self

    def _add_unseen_keys_to_H_dict(self, X):
        """
        Add activations of unseen string categories from X to H_dict.
        """
        unseen_X = np.setdiff1d(X, np.array([*self.H_dict]))
        if unseen_X.size > 0:
            unseen_V = self.ngrams_count.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format='csr')

            unseen_H = _rescale_h(
                unseen_V, np.ones((unseen_V.shape[0], self.n_topics)))
            self._update_H_dict(unseen_X, unseen_H)

    def transform(self, X):
        """
        Transform X using the trained matrix W.

        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, 1)
            The string data to encode.

        Returns
        -------
        X_new : 2-d array, shape (n_samples, n_topics)
            Transformed input.
        """
        X = np.asarray(X)
        assert X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1), f"ERROR:\
        shape {X.shape} of input array is not supported."
        if X.ndim == 2:
            X = X[:, 0]
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "ERROR: Input data is not string."
        unq_X = np.unique(X)
        unq_V = self.ngrams_count.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        for slice in gen_batches(n=unq_H.shape[0],
                                 batch_size=self.batch_size):
            unq_H[slice] = _multiplicative_update_h(
                unq_V[slice], self.W_, unq_H[slice],
                epsilon=1e-3, max_iter=100,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior)
        self._update_H_dict(unq_X, unq_H)
        return self._get_H(X)


def _rescale_W(W, A, B):
    """
    Rescale the topics W to have a L1-norm equal to 1.
    """
    s = W.sum(axis=1, keepdims=True)
    W /= s
    A /= s
    return W, A, B


def _multiplicative_update_w(Vt, W, A, B, Ht, rescale_W, rho):
    """
    Multiplicative update step for the topics W.
    """
    A *= rho
    A += W * safe_sparse_dot(Ht.T, Vt.multiply(np.dot(Ht, W) ** -1))
    B *= rho
    B += Ht.sum(axis=0).reshape(-1, 1)
    np.divide(A, B, out=W)
    if rescale_W:
        _rescale_W(W, A, B)
    return W, A, B


def _rescale_h(V, H):
    """
    Rescale the activations H.
    """
    epsilon = 1e-10  # in case of a document having length=0
    H *= np.maximum(epsilon, V.sum(axis=1).A)
    H /= H.sum(axis=1, keepdims=True)
    return H


def _multiplicative_update_h(Vt, W, Ht, epsilon=1e-3, max_iter=10,
                             rescale_W=False,
                             gamma_shape_prior=1.1, gamma_scale_prior=1.):
    """
    Multiplicative update step for the activations H.
    """
    if rescale_W:
        WT1 = 1 + 1 / gamma_scale_prior
        W_WT1 = W / WT1
    else:
        WT1 = np.sum(W, axis=1) + 1 / gamma_scale_prior
        W_WT1 = W / WT1.reshape(-1, 1)
    const = (gamma_shape_prior - 1) / WT1
    squared_epsilon = epsilon**2
    for vt, ht in zip(Vt, Ht):
        vt_ = vt.data
        idx = vt.indices
        W_WT1_ = W_WT1[:, idx]
        W_ = W[:, idx]
        squared_norm = 1
        for n_iter_ in range(max_iter):
            if squared_norm <= squared_epsilon:
                break
            aux = np.dot(W_WT1_, vt_ / np.dot(ht, W_))
            ht_out = ht * aux + const
            squared_norm = np.dot(
                ht_out - ht, ht_out - ht) / np.dot(ht, ht)
            ht[:] = ht_out
    return Ht


def batch_lookup(lookup, n=1):
    len_iter = len(lookup)
    for idx in range(0, len_iter, n):
        indices = lookup[slice(idx, min(idx + n, len_iter))]
        unq_indices = np.unique(indices)
        yield (unq_indices, indices)


def get_kmeans_prototypes(X, n_prototypes, hashing_dim=128,
                          ngram_range=(2, 4), sparse=False,
                          sample_weight=None, random_state=None):
    """
    Computes prototypes based on:
      - dimensionality reduction (via hashing n-grams)
      - k-means clustering
      - nearest neighbor
    """
    vectorizer = HashingVectorizer(analyzer='char', norm=None,
                                   alternate_sign=False,
                                   ngram_range=ngram_range,
                                   n_features=hashing_dim)
    projected = vectorizer.transform(X)
    if not sparse:
        projected = projected.toarray()
    kmeans = KMeans(n_clusters=n_prototypes, random_state=random_state)
    kmeans.fit(projected, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    neighbors = NearestNeighbors()
    neighbors.fit(projected)
    indexes_prototypes = np.unique(neighbors.kneighbors(centers, 1)[-1])
    return np.sort(X[indexes_prototypes])
