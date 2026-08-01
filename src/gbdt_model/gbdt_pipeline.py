from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier
from src.gbdt_model.transformers import CabinTransformer, SpendingsAdder

categorial_transformer = Pipeline([
    ("to_str_transformer", FunctionTransformer(lambda x: x.astype(str),feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

side_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("to_str_transformer", FunctionTransformer(lambda x: x.astype(str),feature_names_out="one-to-one")),
    ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers = [
    ("categorial", categorial_transformer, ["HomePlanet","CryoSleep", "Destination", "VIP", "Deck"]),
    ("side-transformer", side_transformer, ["Side"])
])


gbdt_model_pipeline = Pipeline([
    ("summarize_spendings", SpendingsAdder()),
    ("deck_extract", CabinTransformer()),
    ("preprocessor", preprocessor),
    ("model", XGBClassifier())   
])