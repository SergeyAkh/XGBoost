import numpy as np
from decision_tree.decision_tree import DecisionTreeRegressor

class MiniXGBoost:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """Train boosting ensemble."""
        y_pred = np.zeros_like(y, dtype=float)

        for i in range(self.n_estimators):
            residuals = y - y_pred
            # print(residuals)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)
            # print(f"Tree {i+1}/{self.n_estimators} fitted. Residual mean: {np.mean(residuals):.4f}")

    def predict(self, X):
        """Sum predictions from all trees."""
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred