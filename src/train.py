import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from src.data_prepare import load_initial_df
from src.log_model.log_pipeline import log_model_pipeline


df = load_initial_df()
X = df.drop(columns=["Transported"])
y = df["Transported"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)

log_model_pipeline.fit(X_train, y_train)

y_pred = log_model_pipeline.predict(X_test)

print("--- Metrics --- ")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:{metrics.f1_score(y_test, y_pred):.4f}")
print(f"ROC, AUC score {metrics.roc_auc_score(y_test, y_pred):.4f}")