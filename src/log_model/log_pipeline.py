from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, SplineTransformer
from src.log_model.transformers import SpendingsAdder, CabinTransformer, NonEssentialDropper

import sklearn
sklearn.set_config(transform_output="pandas")


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])

spline_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("spline", SplineTransformer(n_knots=5, degree=3, knots="quantile"))
])

categorial_transformer = Pipeline([
    ("to_str_transformer", FunctionTransformer(lambda x: x.astype(str),feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("one-hot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
])

side_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("to_str_transformer", FunctionTransformer(lambda x: x.astype(str),feature_names_out="one-to-one")),
    ("one-hot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers = [
    ("numeric", numeric_transformer, ["Spendings"]),
    ("spline", spline_transformer, ["Age"]),
    ("categorial", categorial_transformer, ["HomePlanet","CryoSleep", "Destination", "VIP", "Deck"]),
    ("side-transformer", side_transformer, ["Side"])
])



log_model_pipeline = Pipeline([
    ("dropper", NonEssentialDropper()),
    ("spendings", SpendingsAdder()),
    ("cabin-transformer", CabinTransformer()),
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
