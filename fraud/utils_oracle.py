# ruff: noqa

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class OracleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, pos_label=1):
        self.pos_label = pos_label

    def fit(self, X, y_test):
        self.y_test = y_test
        self.classes_ = np.sort(y_test.unique())
        return self

    def predict(self, X):
        return self.y_test

    def predict_proba(self, X):
        y_proba = (self.y_test == self.pos_label).astype("int32")
        return np.array([1 - y_proba, y_proba]).T
