import pandas as pd
from general_utils.constants import spectral_bands


class Smooth:
    def __init__(self, on=True, overwrite=True):
        self.on = on
        self.overwrite = overwrite

    def avg_smoothing(self, df: pd.DataFrame, window=3) -> pd.DataFrame:
        df = df.sort_values(["id", "time"]).reset_index(drop=True)

        df_smooth = (
            df.groupby("id", group_keys=False)[spectral_bands]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

        if self.overwrite:
            df[spectral_bands] = df_smooth
        else:
            df = pd.concat([df, df_smooth.add_suffix("_smooth")], axis=1)

        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.on:
            return df
        return self.avg_smoothing(df)
