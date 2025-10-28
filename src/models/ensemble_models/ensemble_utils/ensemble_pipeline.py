import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from models.ensemble_models.ensemble_utils.time_series_features import (
    TimeSeriesAggregator,
)


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
            columns=["time", "id", "is_disturbed", "disturbance_year", "date_diff"], errors="ignore"
        )

    def _encode_features(self, df):
        """One-hot encode categorical features."""
        return pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

    def fit(self, train_df):
        ts_builder = TimeSeriesAggregator(window=56, step=28)  # 4 Messpunkte pro Fenster, Schritt 2 Wochen
        feature_df = ts_builder.run(train_df)

        feature_df[self.target_col] = (
            train_df.groupby("id")[self.target_col].first().values
        )

        df = self.drop_columns(feature_df.copy())

        df[self.target_col] = self.label_encoder.fit_transform(df[self.target_col])

        self.categorical_cols = [
            c
            for c in df.select_dtypes(include=["object", "category"]).columns
            if c != self.target_col
        ]

        df = self._encode_features(df)
        X, y = df.drop(columns=[self.target_col]), df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), y

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform().")

        ts_builder = TimeSeriesAggregator(window=56, step=28)  
        feature_df = ts_builder.run(df)

        if self.target_col in df.columns:
            feature_df[self.target_col] = (
                df.groupby("id")[self.target_col].first().values
            )

        df = self.drop_columns(feature_df.copy())

        if self.target_col in df.columns:
            df[self.target_col] = self.label_encoder.transform(df[self.target_col])

        df = self._encode_features(df)
        X = df.drop(columns=[self.target_col]) if self.target_col in df.columns else df
        X = X.reindex(columns=self.scaler.feature_names_in_, fill_value=0)

        X_scaled = self.scaler.transform(X)
        y = df[self.target_col] if self.target_col in df.columns else None

        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return (X_scaled_df, y) if y is not None else X_scaled_df