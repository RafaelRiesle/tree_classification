import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class SITSOutlierCleaner:
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.band_columns = []

    def detect_outliers_isolation_forest(self, df_id):
        df_id = df_id.sort_values("time")

        for band in self.band_columns:
            data = df_id[[band]].values
            data_scaled = StandardScaler().fit_transform(data)

            model = IsolationForest(
                contamination=self.contamination, random_state=self.random_state
            )
            preds = model.fit_predict(data_scaled)

            # Spezifischer Outlier-Flag pro Band
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

        cleaned_df = (
            df.groupby("id", group_keys=False)
            .apply(self.detect_outliers_isolation_forest)
            .pipe(
                lambda d: d.groupby("id", group_keys=False).apply(
                    self.interpolate_outliers
                )
            )
        )
        return cleaned_df
