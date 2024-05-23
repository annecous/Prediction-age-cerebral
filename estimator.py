import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

params = {'alpha': 0.10212637732685018, 'l1_ratio': 0.8415281868893761}

class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, :284]

class VBMFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, 284:]

def get_estimator():
    """Build your estimator here."""
    estimator = make_pipeline(
        VBMFeatureExtractor(),
        StandardScaler(),
        ElasticNet(**params, max_iter=10000)
    )
    return estimator
