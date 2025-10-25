import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
from models.ensemble_models.ensemble_utils.time_series_features import TimeSeriesAggregator


class EnsemblePipeline:
    def __init__(self, target_col, cache_dir=None):
        self.target_col = target_col
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.fitted = False

        # Feature Cache Verzeichnis
        if cache_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            cache_dir = project_root / "data" / "ensemble_training" / "feature_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, split_name):
        return self.cache_dir / f"{split_name}_features.csv"

    def drop_columns(self, df):
        """Remove unnecessary columns."""
        return df.drop(
            columns=["time", "id", "is_disturbed", "disturbance_year", "date_diff"],
            errors="ignore",
        )

    def _encode_features(self, df):
        """One-hot encode categorical features."""
        return pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

    def fit(self, train_df, force_rebuild=False):
        cache_file = self._cache_path("train")
        if cache_file.exists() and not force_rebuild:
            print(f"Loading cached training features from {cache_file}")
            df = pd.read_csv(cache_file)
        else:
            print("Generating new training features...")
            ts_builder = TimeSeriesAggregator(window=56, step=28)
            feature_df = ts_builder.run(train_df)
            feature_df[self.target_col] = train_df.groupby("id")[self.target_col].first().values
            df = self.drop_columns(feature_df)
            df.to_csv(cache_file, index=False)
            print(f"Saved training features to {cache_file}")

        df[self.target_col] = self.label_encoder.fit_transform(df[self.target_col])

        self.categorical_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns if c != self.target_col
        ]

        df = self._encode_features(df)
        X, y = df.drop(columns=[self.target_col]), df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True

        return pd.DataFrame(X_scaled, columns=X.columns), y

    def transform(self, df, split_name="test", force_rebuild=False):
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform().")

        cache_file = self._cache_path(split_name)
        if cache_file.exists() and not force_rebuild:
            print(f"Loading cached {split_name} features from {cache_file}")
            df = pd.read_csv(cache_file)
        else:
            print(f"Generating new {split_name} features...")
            ts_builder = TimeSeriesAggregator(window=56, step=28)
            feature_df = ts_builder.run(df)

            if self.target_col in df.columns:
                feature_df[self.target_col] = df.groupby("id")[self.target_col].first().values

            df = self.drop_columns(feature_df)
            df.to_csv(cache_file, index=False)
            print(f"Saved {split_name} features to {cache_file}")

        if self.target_col in df.columns:
            df[self.target_col] = self.label_encoder.transform(df[self.target_col])

        df = self._encode_features(df)
        X = df.drop(columns=[self.target_col]) if self.target_col in df.columns else df
        X = X.reindex(columns=self.scaler.feature_names_in_, fill_value=0)

        X_scaled = self.scaler.transform(X)
        y = df[self.target_col] if self.target_col in df.columns else None

        return (pd.DataFrame(X_scaled, columns=X.columns), y) if y is not None else pd.DataFrame(X_scaled, columns=X.columns)
