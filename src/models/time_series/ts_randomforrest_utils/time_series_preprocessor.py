import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder


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
        categorical_cols = [
            c
            for c in categorical_cols
            if c in df.columns and c not in self.drop_cols + [self.label_col]
        ]

        if categorical_cols:
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
