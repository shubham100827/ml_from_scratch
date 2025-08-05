import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict the class labels for the input data."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Predict the class label for a single instance."""
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common