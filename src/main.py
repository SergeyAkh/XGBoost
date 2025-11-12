# src/main.py
import numpy as np
import pandas as pd
from data import load_heart_failure_dataset
from XGBoost import MiniXGBoost
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, auc, roc_curve, f1_score


def load_data():
    df, X, y = load_heart_failure_dataset("tan5577/heart-failure-dataset", "HeartDisease")

    return df, X, y




def main():
    df, X, y = load_data()
    model = MiniXGBoost(n_estimators=3, learning_rate=0.2, max_depth=2)
    model.fit(X.values, y.values)
    preds = model.predict(X)
    print("Final predictions:", preds)


if __name__ == "__main__":
    main()


