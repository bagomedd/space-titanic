from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler
from transformers import SpendingsAdder, DeckExtractor, NonEssentialDropper


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])

categorial_transformer = Pipeline([
    ("to_str_transformer", FunctionTransformer(lambda x: x.astype(str))),
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("one-hot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers = [
    ("numeric", numeric_transformer, ["Age","Spendings"]),
    ("categorial", categorial_transformer, ["HomePlanet","CryoSleep", "Destination", "VIP", "Deck"])
])

full_pipeline = Pipeline([
    ("dropper", NonEssentialDropper()),
    ("spendings", SpendingsAdder()),
    ("deck-extractor", DeckExtractor()),
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])