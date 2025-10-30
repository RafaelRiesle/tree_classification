import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class SITSOutlierCleaner:
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.band_columns = []
        self.cleaned_df = None

    def detect_outliers_isolation_forest(self, df_id):
        df_id = df_id.sort_values("time")
        for band in self.band_columns:
            data = df_id[[band]].values
            data_scaled = StandardScaler().fit_transform(data)
            model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=50,
            )
            preds = model.fit_predict(data_scaled)
            df_id[f"is_outlier_{band}"] = preds == -1
        return df_id

    def interpolate_outliers(self, df_id):
        for band in self.band_columns:
            df_id[f"{band}_original"] = df_id[band]
            df_id[band] = np.where(df_id[f"is_outlier_{band}"], np.nan, df_id[band])
            df_id[band] = df_id[band].interpolate(
                method="linear", limit_direction="both"
            )
        return df_id

    def fit_transform(self, df, band_columns):
        self.band_columns = band_columns
        self.cleaned_df = []

        unique_ids = df["id"].unique()
        for uid in tqdm(unique_ids, desc="Processing IDs"):
            df_id = df[df["id"] == uid]
            df_id = self.detect_outliers_isolation_forest(df_id)
            df_id = self.interpolate_outliers(df_id)
            self.cleaned_df.append(df_id)

        self.cleaned_df = pd.concat(self.cleaned_df, ignore_index=True)
        return self.cleaned_df

    def add_any_outlier_flag(self):
        """Add a column 'any_outlier' indicating if any band is an outlier"""
        if self.cleaned_df is None:
            raise ValueError("Please run fit_transform first.")
        outlier_cols = [f"is_outlier_{band}" for band in self.band_columns]
        self.cleaned_df["any_outlier"] = self.cleaned_df[outlier_cols].any(axis=1)
        return self.cleaned_df

    def get_interpolated_only(self):
        """Return only the original band columns plus id and time"""
        if self.cleaned_df is None:
            raise ValueError("Please run fit_transform first.")
        return self.cleaned_df[["id", "time"] + self.band_columns].copy()

    def remaining_outliers_ratio(self):
        """Quote der noch vom IsolationForest markierten Ausreißer"""
        if self.cleaned_df is None:
            raise ValueError("Bitte zuerst fit_transform ausführen.")
        outlier_cols = [f"is_outlier_{band}" for band in self.band_columns]
        remaining = self.cleaned_df[outlier_cols].any(axis=1).sum()
        total = len(self.cleaned_df)
        return remaining / total

    def zscore_outlier_ratio(self, threshold=3):
        if self.cleaned_df is None:
            raise ValueError("Bitte zuerst fit_transform ausführen.")
        z = self.cleaned_df[self.band_columns].apply(zscore)
        mask = (np.abs(z) > threshold).any(axis=1)
        return mask.mean()
