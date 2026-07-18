import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SpendingsAdder(BaseEstimator, TransformerMixin):
    """Summarizes spengings and drops those columns"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        columns = ["ShoppingMall", "RoomService", "FoodCourt", "Spa", "VRDeck"]
        X["Spendings"] = X[columns].sum(axis=1)
        return X.drop(columns=columns)
        
class DeckExtractor(BaseEstimator, TransformerMixin):
    """Extracts Deck letter from cabin"""
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        X = X.assign(
            Deck = lambda X: X["Cabin"].str[0].fillna("Unknown")
        )
        return X.drop(columns=["Cabin"])

class NonEssentialDropper(BaseEstimator, TransformerMixin):
    """Drops Non essential columns"""
    def fit(self, X,y=None):
        return self
    def transform(self, X):
        X = X.copy()
        return X.drop(columns=["PassengerId", "Name"])
    
