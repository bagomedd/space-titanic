
import kagglehub
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from pipeline import full_pipeline

path = kagglehub.competition_download('spaceship-titanic')
sample_submission_df = pd.read_csv(path+"/sample_submission.csv") 
df = pd.read_csv(path+"/train.csv") 

X = df.drop(columns=["Transported"])
y = df["Transported"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")