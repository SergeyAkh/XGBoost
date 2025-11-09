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
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, auc, roc_curve


def load_data():
    df, X, y = load_heart_failure_dataset("tan5577/heart-failure-dataset", "HeartDisease")

    return df, X, y

import importlib

importlib.reload(MiniXGBoost)
df, X, y = load_data()

X = X[["ChestPainType","MaxHR", "ExerciseAngina", "ST_Slope", "Oldpeak"]]

model = MiniXGBoost(n_estimators=100, learning_rate=0.1, max_depth=2, task="classification")

model.fit(X.values, y.values)
probs = model.predict(X.values, as_proba=True)
labels = model.predict(X.values, as_proba=False)

print("Accuracy:", accuracy_score(y, labels))
print("ROC-AUC:", roc_auc_score(y, probs))


def plot_confusion_matrix(Y_true, Y_pred):
    cm = confusion_matrix(Y_true, Y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=target_names, yticklabels=target_names)

    # Set plot labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

def plot_roc_auc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

plot_confusion_matrix(y, labels)
# #########################



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model_sci = XGBClassifier(
    n_estimators=150,
    learning_rate=0.15,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model_sci.fit(X_train, y_train)

# --- Predict and evaluate ---
y_pred = model_sci.predict(X_test)
y_proba = model_sci.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

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


