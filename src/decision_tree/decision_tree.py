import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree_ = None

    # --- impurity ---
    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    # --- split ---
    def _split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    # --- best split ---
    def _best_split(self, X, y):
        best_feat, best_thresh, best_gini = None, None, 1.0
        n_samples, n_features = X.shape

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feat, t)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                g = (len(y_left)*self._gini(y_left) + len(y_right)*self._gini(y_right)) / len(y)
                if g < best_gini:
                    best_feat, best_thresh, best_gini = feat, t, g
        return best_feat, best_thresh

    # --- recursive build ---
    def _build(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return np.bincount(y).argmax()

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return np.bincount(y).argmax()

        X_left, y_left, X_right, y_right = self._split(X, y, feat, thresh)
        left_branch = self._build(X_left, y_left, depth + 1)
        right_branch = self._build(X_right, y_right, depth + 1)

        return (feat, thresh, left_branch, right_branch)

    # --- fit & predict ---
    def fit(self, X, y):
        self.tree_ = self._build(X, y)
        return self

    def _predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feat, thresh, left, right = tree
        branch = left if x[feat] < thresh else right
        return self._predict_one(x, branch)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])