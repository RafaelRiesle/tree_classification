import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1️⃣  DATEN-PREPROCESSOR
# ============================================================


class TimeSeriesPreprocessor:
    """
    Prepares multivariate time series data for RandomForest classification.
    Handles automatic feature selection, encoding, and padding per ID.
    """

    def __init__(
        self,
        feature_cols: List[str] = None,
        label_col: str = "species",
        id_col: str = "id",
    ):
        self.label_col = label_col
        self.id_col = id_col
        self.label_encoder = LabelEncoder()

        # Default columns to drop (id wird erst nach grouping entfernt!)
        self.drop_cols = [
            "time",
            "disturbance_year",
            "is_disturbed",
            "date_diff",
            "year",
        ]

        self.feature_cols = feature_cols

    def load_csv(self, path: str) -> pd.DataFrame:
        """Loads a CSV file and ensures sorting by ID and time."""
        df = pd.read_csv(path)
        if "time" in df.columns:
            df = df.sort_values(by=[self.id_col, "time"])
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encodes categorical variables (e.g. 'season') nur, wenn sie im DataFrame existieren."""
        categorical_cols = ["season", "is_growing_season", "month_num"]
        # Nur Spalten behalten, die auch wirklich im DataFrame sind
        categorical_cols = [
            c
            for c in categorical_cols
            if c in df.columns and c not in self.drop_cols + [self.label_col]
        ]

        if categorical_cols:
            print(f"One-hot encoding categorical columns: {categorical_cols}")
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Determines which columns to use as features."""
        if self.feature_cols is not None:
            return self.feature_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            c for c in numeric_cols if c not in self.drop_cols + [self.label_col]
        ]
        print(f"Auto-detected {len(feature_cols)} feature columns.")
        return feature_cols

    def pad_group(self, group: pd.DataFrame, max_len: int) -> np.ndarray:
        """Pads a single time series group to `max_len` timesteps."""
        data = group[self.feature_cols].to_numpy()
        padded = np.zeros((max_len, len(self.feature_cols)))
        padded[: data.shape[0], :] = data
        return padded

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Groups by ID and converts to padded numpy arrays."""
        df = self._encode_categoricals(df)

        if self.feature_cols is None:
            self.feature_cols = self._select_features(df)

        # Gruppierung nach ID
        grouped = list(df.groupby(self.id_col))
        max_len = max(len(g) for _, g in grouped)

        X = np.stack([self.pad_group(g, max_len) for _, g in grouped])
        y = np.array([g[self.label_col].iloc[0] for _, g in grouped])
        y_encoded = self.label_encoder.fit_transform(y)
        return X, y_encoded

    def prepare_dataset(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads, transforms, and encodes labels for one dataset."""
        df = self.load_csv(csv_path)
        # Drop only non-essential columns (id bleibt!)
        df = df.drop(
            columns=[c for c in self.drop_cols if c in df.columns], errors="ignore"
        )
        return self.transform(df)


# ============================================================
# 2️⃣  DATASET BUILDER
# ============================================================


class PytsDatasetBuilder:
    """Manages train/test/validation dataset preparation."""

    def __init__(
        self,
        train_path: str,
        test_path: str,
        valid_path: str,
        feature_cols: List[str] = None,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.preprocessor = TimeSeriesPreprocessor(feature_cols)

    def build(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Creates X_train, y_train, X_test, y_test, X_valid, y_valid."""
        print("Preparing training data...")
        X_train, y_train = self.preprocessor.prepare_dataset(self.train_path)

        print("Preparing test data...")
        X_test, y_test = self.preprocessor.prepare_dataset(self.test_path)

        print("Preparing validation data...")
        X_valid, y_valid = self.preprocessor.prepare_dataset(self.valid_path)

        return {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
            "valid": (X_valid, y_valid),
        }


# ============================================================
# 3️⃣  MODELLKLASSE (Flatten + RandomForestClassifier)
# ============================================================

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
        scaler: object = StandardScaler(),  # optionaler Scaler
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
            X_flat = X.reshape(n_samples, n_timestamps * n_features)
            return X_flat
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"Unexpected X shape: {X.shape}")

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Flattened scaling: Scaler fit über alle Features."""
        X_flat = self._flatten(X)
        if self.scaler is not None:
            if fit:
                X_flat = self.scaler.fit_transform(X_flat)
            else:
                X_flat = self.scaler.transform(X_flat)
        return X_flat

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the RandomForest model with optional scaling."""
        print("Training RandomForestClassifier (flattened input)...")
        X_flat = self._scale(X_train, fit=True)
        self.model.fit(X_flat, y_train)
        self.is_trained = True
        print("✅ Model trained successfully.")

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict,
        cv: int = 3,
        scoring: str = "accuracy",
    ):
        """Performs GridSearchCV on RandomForest hyperparameters with scaling."""
        print("Running GridSearchCV for RandomForest...")
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
        print(f"✅ Best params: {grid.best_params_}")
        print(f"✅ Best CV score: {grid.best_score_:.4f}")
        return grid.best_params_, grid.best_score_

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, label_encoder=None, set_name: str = "Test"
    ):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        X_flat = self._scale(X, fit=False)
        y_pred = self.model.predict(X_flat)
        acc = accuracy_score(y, y_pred)
        print(f"\n📊 {set_name} Accuracy: {acc:.4f}")

        if label_encoder is not None:
            y_true_labels = label_encoder.inverse_transform(y)
            y_pred_labels = label_encoder.inverse_transform(y_pred)
        else:
            y_true_labels, y_pred_labels = y, y_pred

        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true_labels, y_pred_labels))
        return acc, y_pred

    def save(self, folder: str = "models", label_encoder=None):
        """Saves the trained model + LabelEncoder + Scaler with timestamp."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{folder}/random_forest_{timestamp}.joblib"
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": label_encoder,
                "scaler": self.scaler,
            },
            path,
        )
        print(f"💾 Model + LabelEncoder + Scaler saved to: {path}")
        return path

    def load(self, path: str):
        """Loads a trained model + LabelEncoder + Scaler."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data.get("scaler", None)
        self.is_trained = True
        label_encoder = data.get("label_encoder", None)
        print(f"📦 Model loaded from: {path}")
        return label_encoder


# ============================================================
# 4️⃣  ANWENDUNGSBEISPIEL
# ============================================================

if __name__ == "__main__":
    # ---------------------------
    # Pfade
    # ---------------------------
    train_path = "../../../data/processed/trainset.csv"
    test_path = "../../../data/processed/testset.csv"
    valid_path = "../../../data/processed/valset.csv"

    # ---------------------------
    # Daten vorbereiten
    # ---------------------------
    dataset_builder = PytsDatasetBuilder(train_path, test_path, valid_path)
    datasets = dataset_builder.build()

    X_train, y_train = datasets["train"]
    X_test, y_test = datasets["test"]
    X_valid, y_valid = datasets["valid"]

    label_encoder = dataset_builder.preprocessor.label_encoder

    # ---------------------------
    # Modell instanziieren
    # ---------------------------
    model = FlattenedRandomForestModel(random_state=42)

    # ---------------------------
    # GridSearch Hyperparameter
    # ---------------------------
    param_grid = {
        "n_estimators": [30],
    }

    # ---------------------------
    # GridSearch durchführen
    # ---------------------------
    best_params, best_score = model.grid_search(
        X_train, y_train, param_grid=param_grid, cv=3, scoring="accuracy"
    )
    print(f"Beste Hyperparameter: {best_params}, Bestes CV-Score: {best_score:.4f}")

    # ---------------------------
    # Evaluation
    # ---------------------------
    model.evaluate(X_test, y_test, label_encoder=label_encoder, set_name="Test")
    model.evaluate(X_valid, y_valid, label_encoder=label_encoder, set_name="Validation")

    # ---------------------------
    # Modell + LabelEncoder speichern (mit Timestamp)
    # ---------------------------
    save_path = model.save(
        folder="../../../data/pyts/models", label_encoder=label_encoder
    )

    # ---------------------------
    # Modell wieder laden
    # ---------------------------
    model2 = FlattenedRandomForestModel()
    loaded_label_encoder = model2.load(save_path)

    # Optional: Testen, ob das geladene Modell funktioniert
    model2.evaluate(
        X_test, y_test, label_encoder=loaded_label_encoder, set_name="Test (loaded)"
    )
