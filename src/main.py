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

df, X, y = load_data()

X = X[["ChestPainType","MaxHR", "ExerciseAngina", "ST_Slope", "Oldpeak"]]

model = MiniXGBoost(n_estimators=200, learning_rate=0.1, max_depth=5, task="classification")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train.values, y_train.values)
y_test_probs = model.predict(X_test.values, as_proba=True)
y_test_labels = model.predict(X_test.values, as_proba=False)

print("Accuracy:", accuracy_score(y_test, y_test_labels))
print("ROC-AUC:", roc_auc_score(y_test, y_test_probs))

thresholds = np.linspace(0, 1, 100)
accuracies = [accuracy_score(y_test, y_test_probs >= t) for t in thresholds]
precisions = [precision_score(y_test, y_test_probs >= t, zero_division=0) for t in thresholds]
recalls = [recall_score(y_test, y_test_probs >= t) for t in thresholds]

best_t = thresholds[np.argmax(accuracies)]
best_acc = np.max(accuracies)
f1 = f1_score(y_test, y_test_probs >= best_t)
print("Best threshold:", best_t)
print("Best accuracy:", best_acc)
print("F1 score:", f1)

# 5️⃣ Plot results
plt.figure(figsize=(8,5))
plt.plot(thresholds, accuracies, label='Accuracy', lw=2)
plt.plot(thresholds, precisions, label='Precision', linestyle='--')
plt.plot(thresholds, recalls, label='Recall', linestyle=':')
plt.axvline(best_t, color='r', linestyle='--', label=f'Best Threshold = {best_t:.2f}')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Metrics vs. Decision Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


target_names = ["normal", "abnormal"]
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
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

plot_confusion_matrix(y_test, y_test_labels)


def main():
    df, X, y = load_data()
    model = MiniXGBoost(n_estimators=3, learning_rate=0.2, max_depth=2)
    model.fit(X.values, y.values)
    preds = model.predict(X)
    print("Final predictions:", preds)


if __name__ == "__main__":
    main()


