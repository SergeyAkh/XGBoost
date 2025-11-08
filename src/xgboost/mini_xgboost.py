import numpy as np
from decision_tree.decision_tree import DecisionTreeRegressor

class MiniXGBoost:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3, task="regression"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.task = task

    def _sigmoid(self, x):
        """Numerically stable sigmoid"""
        # x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """Train boosting ensemble."""
        y_pred = np.zeros_like(y, dtype=float)

        for i in range(self.n_estimators):
            if self.task == "classification":
                y_pred_proba = self._sigmoid(y_pred)
                residuals = y - y_pred_proba
            else:
                residuals = y - y_pred

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X, as_proba=False):
        """Sum predictions from all trees."""
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        if self.task == "classification":
            y_pred = self._sigmoid(y_pred)
            if not as_proba:
                y_pred = (y_pred >= 0.5).astype(int)

        return y_pred
