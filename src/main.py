# src/main.py
import numpy as np
from decision_tree.decision_tree import DecisionTreeClassifier


def load_data():
    """Generate or load sample data."""
    X = np.array([
        [2.7, 2.5],
        [1.3, 3.0],
        [3.1, 1.2],
        [0.8, 1.5],
        [3.0, 3.5]
    ])
    y = np.array([0, 0, 1, 0, 1])
    return X, y


def train_tree():
    """Train and evaluate the decision tree."""
    X, y = load_data()

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    print("Tree trained successfully.")
    print("Predictions:", tree.predict(X))


def main():
    """Main entry point."""
    train_tree()


if __name__ == "__main__":
    main()
