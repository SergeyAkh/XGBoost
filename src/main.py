# src/main.py
import numpy as np
import pandas as pd

from data import load_heart_failure_dataset
from xgboost.mini_xgboost import MiniXGBoost


import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    df, X, y = load_heart_failure_dataset("tan5577/heart-failure-dataset", "HeartDisease")

    return df, X, y


df, X, y = load_data()


df = pd.get_dummies(df, columns=["ChestPainType", "Sex", "RestingECG", "ExerciseAngina", "ST_Slope"], drop_first=False)


X["HeartDisease"] = y

corr = X.corr(numeric_only=True)
target_corr = X["HeartDisease"].sort_values(ascending=False)
print(corr)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix (Features + Target)")
plt.show()

X.drop("HeartDisease", axis=1, inplace=True)

X = X[["ChestPainType","MaxHR", "ExerciseAngina", "ST_Slope", "Oldpeak"]]

model = MiniXGBoost(n_estimators=3, learning_rate=0.2, max_depth=2)


print(model.trees)
model.fit(X.values, y.values)
preds = model.predict(X)


def main():
    df, X, y = load_data()
    model = MiniXGBoost(n_estimators=3, learning_rate=0.2, max_depth=2)
    model.fit(X.values, y.values)
    preds = model.predict(X)
    print("Final predictions:", preds)


if __name__ == "__main__":
    main()


