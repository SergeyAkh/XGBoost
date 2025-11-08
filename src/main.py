# src/main.py
import numpy as np
import pandas as pd

from data import load_heart_failure_dataset
from XGBoost.mini_xgboost import MiniXGBoost


import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    df, X, y = load_heart_failure_dataset("tan5577/heart-failure-dataset", "HeartDisease")

    return df, X, y


df, X, y = load_data()

X = X[["ChestPainType","MaxHR", "ExerciseAngina", "ST_Slope", "Oldpeak"]]

model = MiniXGBoost(n_estimators=10, learning_rate=0.1, max_depth=2, task="classification")

model.fit(X.values, y.values)
probs = model.predict(X.values, as_proba=True)

# #########################

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

model_sci = XGBClassifier(
    n_estimators=5,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
model_sci.fit(X, y)

# --- Predict and evaluate ---
y_pred = model_sci.predict(X)
y_proba = model_sci.predict_proba(X)[:, 1]

print(y_proba)

##########################



print(X.head())

def main():
    df, X, y = load_data()
    model = MiniXGBoost(n_estimators=3, learning_rate=0.2, max_depth=2)
    model.fit(X.values, y.values)
    preds = model.predict(X)
    print("Final predictions:", preds)


if __name__ == "__main__":
    main()


