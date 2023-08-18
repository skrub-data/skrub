"""
Implements the Frequency Encoder, a transformer that allows
encoding a feature using it's frequency.
"""



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        column, 
        bins
    ):
        
        self.column = column
        self.bins = bins
        self.uniques_to_map = None

        
       

    def fit(self, X:pd.DataFrame,  y=None):
        value_counts_series = X[self.column].value_counts()
        self.uniques_to_map = pd.cut(value_counts_series, self.bins, right=False)

        return self

    
    def transform(self, X) -> pd.DataFrame:
        return X[self.column].map(self.uniques_to_map)
