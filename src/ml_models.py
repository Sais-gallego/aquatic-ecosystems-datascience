"""
Machine Learning Forecasting & Classification for Aquatic Environments.
DATA20067 - Data Science for Monitoring Aquatic Ecosystems, University of Helsinki.
"""

import math
from typing import List, Tuple, Dict, Any, Optional


class AquaticMetrics:
    """Calculates regression and classification evaluation metrics."""

    @staticmethod
    def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
        if not y_true:
            return 0.0
        mse = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
        return math.sqrt(mse)

    @staticmethod
    def r2_score(y_true: List[float], y_pred: List[float]) -> float:
        if len(y_true) < 2:
            return 0.0
        y_mean = sum(y_true) / len(y_true)
        ss_tot = sum((yt - y_mean)**2 for yt in y_true)
        ss_res = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred))
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @staticmethod
    def classification_f1(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if y_true else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }


class ChlorophyllForecaster:
    """
    Multivariate Ridge Regression regressor for Chlorophyll-a forecasting.
    Includes feature standardization and closed-form / gradient descent solver.
    """

    def __init__(self, l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.means: List[float] = []
        self.stds: List[float] = []

    def fit(self, X: List[List[float]], y: List[float], epochs: int = 40, lr: float = 0.05):
        n = len(X)
        if n == 0:
            return
        num_feats = len(X[0])

        # Standardize features
        self.means = [sum(X[i][j] for i in range(n)) / n for j in range(num_feats)]
        self.stds = []
        for j in range(num_feats):
            var = sum((X[i][j] - self.means[j])**2 for i in range(n)) / (n - 1 if n > 1 else 1)
            std = math.sqrt(var)
            self.stds.append(std if std > 1e-9 else 1.0)

        X_norm = [[(X[i][j] - self.means[j]) / self.stds[j] for j in range(num_feats)] for i in range(n)]

        self.weights = [0.0] * num_feats
        self.bias = sum(y) / n

        for _ in range(epochs):
            for i in range(n):
                pred = sum(self.weights[j] * X_norm[i][j] for j in range(num_feats)) + self.bias
                err = pred - y[i]
                for j in range(num_feats):
                    self.weights[j] -= lr * (err * X_norm[i][j] + (self.l2_reg / n) * self.weights[j])
                self.bias -= lr * err

    def predict(self, X: List[List[float]]) -> List[float]:
        num_feats = len(self.weights)
        preds = []
        for row in X:
            norm_row = [(row[j] - self.means[j]) / self.stds[j] for j in range(num_feats)]
            p = sum(self.weights[j] * norm_row[j] for j in range(num_feats)) + self.bias
            preds.append(max(0.0, p))  # Chlorophyll-a concentration cannot be negative
        return preds


class AlgaeBloomClassifier:
    """
    Logistic Regression classifier detecting Harmful Algal Bloom (HAB) risk (Chl-a >= 20 ug/L).
    """

    def __init__(self, bloom_threshold_ug_l: float = 20.0):
        self.threshold = bloom_threshold_ug_l
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.means: List[float] = []
        self.stds: List[float] = []

    def fit(self, X: List[List[float]], y_chla: List[float], epochs: int = 50, lr: float = 0.2):
        # Binarize target
        y_binary = [1 if val >= self.threshold else 0 for val in y_chla]
        n = len(X)
        if n == 0:
            return
        num_feats = len(X[0])

        # Standardize features
        self.means = [sum(X[i][j] for i in range(n)) / n for j in range(num_feats)]
        self.stds = []
        for j in range(num_feats):
            var = sum((X[i][j] - self.means[j])**2 for i in range(n)) / (n - 1 if n > 1 else 1)
            std = math.sqrt(var)
            self.stds.append(std if std > 1e-9 else 1.0)

        X_norm = [[(X[i][j] - self.means[j]) / self.stds[j] for j in range(num_feats)] for i in range(n)]

        self.weights = [0.0] * num_feats
        self.bias = 0.0

        for _ in range(epochs):
            for i in range(n):
                z = sum(self.weights[j] * X_norm[i][j] for j in range(num_feats)) + self.bias
                z_clamped = max(-20.0, min(20.0, z))
                p = 1.0 / (1.0 + math.exp(-z_clamped))
                err = p - y_binary[i]
                for j in range(num_feats):
                    self.weights[j] -= lr * (err * X_norm[i][j] / n)
                self.bias -= lr * (err / n)

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        num_feats = len(self.weights)
        probs = []
        for row in X:
            norm_row = [(row[j] - self.means[j]) / self.stds[j] for j in range(num_feats)]
            z = sum(self.weights[j] * norm_row[j] for j in range(num_feats)) + self.bias
            z_clamped = max(-20.0, min(20.0, z))
            probs.append(1.0 / (1.0 + math.exp(-z_clamped)))
        return probs

    def predict(self, X: List[List[float]], cutoff: float = 0.5) -> List[int]:
        return [1 if p >= cutoff else 0 for p in self.predict_proba(X)]
