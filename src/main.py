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

corr = df.corr(numeric_only=True)
target_corr = corr["HeartDisease"].sort_values(ascending=False)


plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix (Features + Target)")
plt.show()

def main():
    df, X, y = load_data()
    model = MiniXGBoost(n_estimators=3, learning_rate=0.2, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    print("Final predictions:", preds)


if __name__ == "__main__":
    main()


