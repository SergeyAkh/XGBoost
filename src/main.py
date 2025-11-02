# src/main.py
import numpy as np
from xgboost.mini_xgboost import MiniXGBoost

def load_data():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ])
    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    return X, y

print(load_data())
def main():
    X, y = load_data()
    model = MiniXGBoost(n_estimators=3, learning_rate=0.2, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    print("Final predictions:", preds)


if __name__ == "__main__":
    main()

X, y = load_data()
model = MiniXGBoost(n_estimators=3, learning_rate=0.2, max_depth=2)
model.fit(X, y)
preds = model.predict(X)

