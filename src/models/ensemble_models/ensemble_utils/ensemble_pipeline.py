import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class EnsemblePipeline:
    def __init__(self, target_col):
        self.target_col = target_col
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.fitted = False

    def drop_columns(self, df):
        """Remove unnecessary columns."""
        return df.drop(
            columns=["time", "id", "disturbed", "disturbance_year"], errors="ignore"
        )

    def _encode_features(self, df):
        """One-hot encode categorical features."""
        return pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

    def fit(self, train_df):
        df = self.drop_columns(train_df.copy())

        # Encode target
        df[self.target_col] = self.label_encoder.fit_transform(df[self.target_col])

        # Identify categorical columns
        self.categorical_cols = [
            c
            for c in df.select_dtypes(include=["object", "category"]).columns
            if c != self.target_col
        ]

        # One-hot encode and split
        df = self._encode_features(df)
        X, y = df.drop(columns=[self.target_col]), df[self.target_col]

        # Scale
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), y

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform().")

        df = self.drop_columns(df.copy())

        if self.target_col in df.columns:
            df[self.target_col] = self.label_encoder.transform(df[self.target_col])

        df = self._encode_features(df)

        # Reorder columns to match training set
        X = df.drop(columns=[self.target_col]) if self.target_col in df.columns else df
        X = X.reindex(columns=self.scaler.feature_names_in_, fill_value=0)

        X_scaled = self.scaler.transform(X)
        y = df[self.target_col] if self.target_col in df.columns else None

        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return (X_scaled_df, y) if y is not None else X_scaled_df
