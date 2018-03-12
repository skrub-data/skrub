"""
Encodings based on n-grams.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse


def get_unique_ngrams(string, n):
    """ Return the set of different n-grams in a string
    """
    # string = '$$' + string + '$$'
    strings = [string[i:] for i in range(n)]
    return set(zip(*strings))


def get_ngrams(string, n):
    """ Return a list with all n-grams in a string.
    """
    # string = '$$' + string + '$$'
    strings = [string[i:] for i in range(n)]
    return list(zip(*strings))


def ngram_similarity(X, cats, n, sim_type=None, dtype=np.float64):
    """ given to arrays of strings, returns the
    similarity encoding matrix of size
    len(X) x len(cats)
    sim1(s_i, s_j) = ||min(ci, cj)||_1 /
                    (||ci||_1 + ||cj||_1 - ||min(ci, cj)||_1)

    sim2(s_i, s_j) = 2||min(ci, cj)||_1/ (||ci||_1 + ||cj||_1)

    sim2_1(s_i, s_j) = 2 dot(c1, c2) / (dot(c1, c1) + dot(c2, c2))

    sim2_2(s_i, s_j) = 2 dot(p1, p2) / (dot(p1, p1) + dot(p2, p2))

    sim3(s_i, s_j) = dot(ci, cj) / (dot(ci, ci) + dot(cj, cj) - dot(ci, cj))

    sim3_2(s_i, s_j) = dot(p1, p2) / (dot(p1, p1) + dot(p2, p2) - dot(p1, p2))

    sim4(s_i, s_j) = dot(c1, c2) / (dot(c1, c1)^.5 * dot(c2, c2)^.5)

    sim5(s_i, s_j) = dot(p1, p2) / (dot(p1, p1)^.5 * dot(p2, p2)^.5)

    sim6(s_i, s_j) = dot(c1, c2)

    sim7(s_i, s_j) = dot(p1, p2)

    fisher_kernel
    """
    if sim_type == 'sim1':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        count2 = vectorizer.fit_transform(cats)
        count1 = vectorizer.transform(unq_X)
        sum2 = count2.sum(axis=1)
        SE_dict = {}
        for i, x in enumerate(count1):
            aux = sparse.csr_matrix(np.ones((count2.shape[0], 1))).dot(x)
            samegrams = count2.minimum(aux).sum(axis=1)
            allgrams = x.sum() + sum2 - samegrams
            similarity = np.divide(samegrams, allgrams)
            SE_dict[unq_X[i]] = np.array(similarity).reshape(-1)
        SE = []
        for x in X:
            SE.append(SE_dict[x])
        return np.nan_to_num(np.vstack(SE))

    if sim_type == 'sim2':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        count2 = vectorizer.fit_transform(cats)
        count1 = vectorizer.transform(unq_X)
        sum_matrix2 = count2.sum(axis=1)
        SE_dict = {}
        for i, x in enumerate(count1):
            aux = sparse.csr_matrix(np.ones((count2.shape[0], 1))).dot(x)
            samegrams = count2.minimum(aux).sum(axis=1)
            allgrams = x.sum() + sum_matrix2
            similarity = 2 * np.divide(samegrams, allgrams)
            SE_dict[unq_X[i]] = np.array(similarity).reshape(-1)
        SE = []
        for x in X:
            SE.append(SE_dict[x])
        return np.nan_to_num(np.vstack(SE))

    if sim_type == 'sim2_1':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = vectorizer.fit_transform(cats).transpose()
        Ci = vectorizer.transform(unq_X)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                      (1, Cj.shape[1]))
        cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                      (Ci.shape[0], 1))
        similarity = np.divide(2*cij, cii + cjj)
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[s] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'sim2_2':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = (vectorizer.fit_transform(cats) > 0
              ).astype(dtype).transpose()
        Ci = (vectorizer.transform(unq_X) > 0).astype(dtype)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                      (1, Cj.shape[1]))
        cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                      (Ci.shape[0], 1))
        similarity = np.divide(2*cij, cii + cjj)
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'sim3':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = vectorizer.fit_transform(cats).transpose()
        Ci = vectorizer.transform(unq_X)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                      (1, Cj.shape[1]))
        cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                      (Ci.shape[0], 1))
        similarity = np.divide(cij, cii + cjj - cij)
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'sim3_2':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = (vectorizer.fit_transform(cats) > 0
              ).astype(dtype).transpose()
        Ci = (vectorizer.transform(unq_X) > 0).astype(dtype)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                      (1, Cj.shape[1]))
        cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                      (Ci.shape[0], 1))
        similarity = np.divide(cij, cii + cjj - cij)
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'sim4':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = vectorizer.fit_transform(cats).transpose()
        Ci = vectorizer.transform(unq_X)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(np.power(Ci.multiply(Ci).sum(axis=1), .5),
                      (1, Cj.shape[1]))
        cjj = np.tile(np.power(Cj.multiply(Cj).sum(axis=0), .5),
                      (Ci.shape[0], 1))
        similarity = np.divide(cij, np.multiply(cii, cjj))
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'sim5':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = (vectorizer.fit_transform(cats) > 0
              ).astype(dtype).transpose()
        Ci = (vectorizer.transform(unq_X) > 0).astype(dtype)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(np.power(Ci.multiply(Ci).sum(axis=1), .5),
                      (1, Cj.shape[1]))
        cjj = np.tile(np.power(Cj.multiply(Cj).sum(axis=0), .5),
                      (Ci.shape[0], 1))
        similarity = np.divide(cij, np.multiply(cii, cjj))
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'sim6':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = vectorizer.fit_transform(cats).transpose()
        Ci = vectorizer.transform(unq_X)
        SE_dict = {}
        similarity = Ci.dot(Cj).toarray()
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'sim7':
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = (vectorizer.fit_transform(cats) > 0
              ).astype(dtype).transpose()
        Ci = (vectorizer.transform(unq_X) > 0).astype(dtype)
        SE_dict = {}
        similarity = Ci.dot(Cj).toarray()
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'fisher_kernel':
        unq_X = np.unique(X)
        unq_cats, count_j = np.unique(cats, return_counts=True)
        theta = count_j/sum(count_j)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = (vectorizer.fit_transform(unq_cats) > 0
              ).astype(dtype).toarray()
        Ci = (vectorizer.transform(unq_X) > 0).astype(dtype).toarray()
        m = Cj.shape[1]
        SE_dict = {}
        for i, p_i in enumerate(Ci):
            gamma = np.zeros(m)
            for j, p_j in enumerate(Cj):
                gamma += (p_j == p_i).astype(dtype)*theta[j]
            similarity = []
            for j, p_j in enumerate(Cj):
                sim_j = (p_j == p_i).astype(dtype) / gamma
                similarity.append(sim_j.sum())
            SE_dict[unq_X[i]] = np.array(similarity)
        SE = []
        for x in X:
            SE.append(SE_dict[x])
        return np.nan_to_num(np.vstack(SE))


def ngrams_count_vectorizer(strings, n):
    """ Return the a disctionary with the count of every
    unique n-gram in the string.
    """
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    count = vectorizer.fit_transform(strings)
    feature_names = vectorizer.get_feature_names()
    return count, feature_names


def ngrams_hashing_vectorizer(strings, n, n_features):
    """ Return the a disctionary with the count of every
    unique n-gram in the string.
    """
    hv = HashingVectorizer(analyzer='char', ngram_range=(n, n),
                           n_features=n_features, norm=None,
                           alternate_sign=False)
    hash_matrix = hv.fit_transform(strings)
    return hash_matrix
