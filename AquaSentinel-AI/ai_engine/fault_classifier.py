"""
AquaSentinel AI - Fault Classification Engine
Supervised classification using Random Forest to distinguish between normal hydraulic
operations and physical leak events, with evaluation strictly on held-out test data.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class FaultClassifier:
    """
    Random Forest classifier for hydraulic event discrimination (Normal vs. Leak).
    """

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.classes_: List[int] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "FaultClassifier":
        """
        Trains the Random Forest classifier on training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Hydraulic features (pressures, flows).
        y_train : pd.Series
            Binary labels (0: Normal, 1: Leak).
        """
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts binary class (0: Normal, 1: Leak) for each sample.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X[self.feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts posterior probabilities [P(normal), P(leak)].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X[self.feature_names])

    def evaluate_test_set(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluates classifier performance strictly on the held-out TEST set.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing accuracy, precision, recall, f1,
            confusion matrix, classification report, and top feature importances.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        y_true = np.array(y_test)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Leak"],
            output_dict=True,
            zero_division=0,
        )

        # Extract top feature importances
        importances = self.model.feature_importances_
        feature_rankings = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        top_features = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in feature_rankings[:8]
        ]

        return {
            "test_sample_count": len(y_test),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "top_features": top_features,
        }
