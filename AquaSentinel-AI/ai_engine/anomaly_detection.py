"""
AquaSentinel AI - Hydraulic Anomaly Detection Engine
Unsupervised anomaly detection using Isolation Forest trained strictly on normal
operating conditions to detect hydraulic deviations and leaks.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class HydraulicAnomalyDetector:
    """
    Unsupervised Isolation Forest detector trained exclusively on normal hydraulic data.
    Flags physical anomalies and yields continuous anomaly scores.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            bootstrap=False,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.feature_names = []
        self._score_min = -0.5
        self._score_max = 0.5

    def fit(self, X_normal: pd.DataFrame) -> "HydraulicAnomalyDetector":
        """
        Trains the Isolation Forest on NORMAL baseline data only.

        Parameters
        ----------
        X_normal : pd.DataFrame
            Hydraulic sensor observations under non-leak, normal operating conditions.
        """
        if len(X_normal) == 0:
            raise ValueError("Training dataset X_normal cannot be empty.")

        self.feature_names = list(X_normal.columns)
        X_scaled = self.scaler.fit_transform(X_normal)
        self.model.fit(X_scaled)

        # Baseline offset for score calibration
        raw_decisions = self.model.decision_function(X_scaled)
        self._score_min = float(np.percentile(raw_decisions, 1))
        self._score_max = float(np.percentile(raw_decisions, 99))
        self.is_fitted = True
        return self

    def compute_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes a continuous anomaly score in [0.0, 1.0] for each sample,
        where 0.0 = completely normal and 1.0 = highly anomalous.

        Parameters
        ----------
        X : pd.DataFrame
            Sensor measurement features.

        Returns
        -------
        np.ndarray
            1D array of calibrated anomaly scores in range [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing anomaly scores.")

        X_scaled = self.scaler.transform(X[self.feature_names])
        raw_decision = self.model.decision_function(X_scaled)

        # Sigmoid calibration: lower decision function = higher anomaly score
        # decision_function: positive for inliers, negative for outliers
        scale_range = max(self._score_max - self._score_min, 1e-6)
        normalized = (raw_decision - self._score_min) / scale_range
        # Invert so 1 is anomalous and 0 is normal, bounded in [0, 1]
        scores = 1.0 / (1.0 + np.exp(6.0 * normalized))
        return np.clip(scores, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts binary anomaly state: 0 for normal, 1 for anomaly.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        X_scaled = self.scaler.transform(X[self.feature_names])
        raw_preds = self.model.predict(X_scaled)
        # Sklearn maps 1 -> inlier (normal = 0), -1 -> outlier (anomaly = 1)
        binary_preds = np.where(raw_preds == -1, 1, 0)
        return binary_preds

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the detector against a held-out test set containing both
        normal (0) and anomalous/leak (1) scenarios.
        """
        y_true = np.array(y_test)
        y_pred = self.predict(X_test)
        scores = self.compute_anomaly_score(X_test)

        normal_mask = (y_true == 0)
        leak_mask = (y_true == 1)

        auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 1.0
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "roc_auc": round(float(auc), 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "mean_anomaly_score_normal": round(float(scores[normal_mask].mean()), 4) if normal_mask.any() else 0.0,
            "mean_anomaly_score_leak": round(float(scores[leak_mask].mean()), 4) if leak_mask.any() else 0.0,
            "confusion_matrix": cm.tolist(),
        }
