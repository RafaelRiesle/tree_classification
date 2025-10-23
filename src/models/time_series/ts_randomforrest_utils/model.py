import os
from datetime import datetime
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class FlattenedRandomForestModel:
    """
    A RandomForestClassifier wrapper that flattens multivariate time series.
    Supports optional feature scaling, hyperparameter tuning, and timestamped saving.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        scaler: object = StandardScaler(),
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.scaler = scaler
        self.is_trained = False

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            n_samples, n_timestamps, n_features = X.shape
            return X.reshape(n_samples, n_timestamps * n_features)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"Unexpected X shape: {X.shape}")

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X_flat = self._flatten(X)
        if self.scaler is not None:
            if fit:
                X_flat = self.scaler.fit_transform(X_flat)
            else:
                X_flat = self.scaler.transform(X_flat)
        return X_flat

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        X_flat = self._scale(X_train, fit=True)
        self.model.fit(X_flat, y_train)
        self.is_trained = True

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict,
        cv: int = 3,
        scoring: str = "accuracy",
    ):
        X_flat = self._scale(X_train, fit=True)
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=2,
        )
        grid.fit(X_flat, y_train)
        self.model = grid.best_estimator_
        self.is_trained = True
        return grid.best_params_, grid.best_score_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        X_flat = self._scale(X, fit=False)
        return self.model.predict(X_flat)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, label_encoder=None, set_name: str = "Test"
    ):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        X_flat = self._scale(X, fit=False)
        y_pred = self.model.predict(X_flat)
        acc = accuracy_score(y, y_pred)
        print(f"{set_name} Accuracy: {acc:.4f}")

        if label_encoder is not None:
            y_true_labels = label_encoder.inverse_transform(y)
            y_pred_labels = label_encoder.inverse_transform(y_pred)
        else:
            y_true_labels, y_pred_labels = y, y_pred

        print("Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true_labels, y_pred_labels))

        # Weighted accuracy based on F1-score
        report = classification_report(y, y_pred, output_dict=True)
        weighted_acc = report["weighted avg"]["f1-score"]
        print(f"Weighted Accuracy (F1, {set_name}): {weighted_acc:.4f}")

        return acc, y_pred

    def save(self, folder: str = "models", label_encoder=None):
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{folder}/random_forest_{timestamp}.joblib"
        joblib.dump(
            {"model": self.model, "label_encoder": label_encoder, "scaler": self.scaler},
            path,
        )
        return path

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data.get("scaler", None)
        self.is_trained = True
        return data.get("label_encoder", None)