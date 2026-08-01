import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from src.data_prepare import load_initial_df
from src.gbdt_model.gbdt_pipeline import gbdt_model_pipeline


df = load_initial_df()
X = df.drop(columns=["Transported"])
y = df["Transported"]

X_train_valid, X_test, y_train_valid, y_test = train_test_split(X,y, test_size=0.15, random_state=42)
# X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.15, random_state=42)

search = GridSearchCV(
    estimator=gbdt_model_pipeline,
    param_grid={
        "model__max_depth": [2,3,4,5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "model__n_estimators": [50,100,150,200,250]
    },
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)
search.fit(X_train_valid, y_train_valid)

# gbdt_model_pipeline.fit(X_train, y_train) Значит это уже не нужно?

# y_pred = gbdt_model_pipeline.predict(X_test) А вместо этого
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("--- Hyperparams ---")
print(search.best_params_)

print("--- Metrics --- ")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:{metrics.f1_score(y_test, y_pred):.4f}")
print(f"ROC, AUC score {metrics.roc_auc_score(y_test, y_pred):.4f}")