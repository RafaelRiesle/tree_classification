import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class TimeSeriesPreprocessor:
    """
    Prepares multivariate time series data for RandomForest classification.
    Handles automatic feature selection, encoding, and padding per ID.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "species",
        id_col: str = "id",
        fixed_seq_len: int = 26,
    ):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.id_col = id_col
        self.max_len = fixed_seq_len

        self.label_encoder = LabelEncoder()
        self.ohe: Optional[OneHotEncoder] = None
        self.max_len: Optional[int] = None

        self.drop_cols = [
            "time",
            "disturbance_year",
            "doy",
            "is_disturbed",
            "date_diff",
            "year",
        ]

    # -----------------------------------------------------
    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "time" in df.columns:
            df = df.sort_values(by=[self.id_col, "time"])
        return df

    # -----------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """One-hot encodes categorical variables using sklearn's OneHotEncoder."""
        categorical_cols = [
            c
            for c in [
                "season",
                "is_growing_season",
                "month_num",
                "week_of_year",
                "biweek_of year",
            ]
            if c in df.columns and c not in self.drop_cols + [self.label_col]
        ]

        if not categorical_cols:
            return df

        if fit or self.ohe is None:
            self.ohe = OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            )
            encoded = self.ohe.fit_transform(df[categorical_cols])
        else:
            encoded = self.ohe.transform(df[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=self.ohe.get_feature_names_out(categorical_cols),
            index=df.index,
        )

        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        return df

    # -----------------------------------------------------
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        if self.feature_cols is not None:
            return self.feature_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            c
            for c in numeric_cols
            if c not in self.drop_cols + [self.label_col, self.id_col]
        ]
        print(
            f"Auto-detected {len(feature_cols)} feature columns (excluded '{self.id_col}')."
        )
        return feature_cols

    # -----------------------------------------------------
    def pad_group(self, group: pd.DataFrame) -> np.ndarray:
        data = group[self.feature_cols].to_numpy()
        padded = np.zeros((self.max_len, len(self.feature_cols)))
        length = min(data.shape[0], self.max_len)
        padded[:length, :] = data[:length, :]
        return padded

    # -----------------------------------------------------
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = self._encode_categoricals(df, fit=False)

        if self.feature_cols is None:
            self.feature_cols = self._select_features(df)

        grouped = list(df.groupby(self.id_col))

        X = np.stack([self.pad_group(g) for _, g in grouped])
        y = np.array([g[self.label_col].iloc[0] for _, g in grouped])
        y_encoded = self.label_encoder.transform(y)
        return X, y_encoded

    # -----------------------------------------------------
    def prepare_dataset(
        self, csv_path: str, fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads, encodes, and transforms dataset. If `fit=True`, fits encoders & sets max_len."""
        df = self.load_csv(csv_path)
        df = df.drop(
            columns=[c for c in self.drop_cols if c in df.columns], errors="ignore"
        )

        # Fit encoder and determine max_len only on training data
        df = self._encode_categoricals(df, fit=fit)

        if fit:
            self.feature_cols = self._select_features(df)
            grouped = list(df.groupby(self.id_col))
            self.max_len = max(len(g) for _, g in grouped)
            # Fit label encoder
            y = np.array([g[self.label_col].iloc[0] for _, g in grouped])
            self.label_encoder.fit(y)

        return self.transform(df)
