import pandas as pd
from general_utils.constants import spectral_bands

class Smooth:
    def __init__(self, on=True):
        self.on = on

    def avg_smoothing(self, df: pd.DataFrame, window=3) -> pd.DataFrame:
        """Smoothing Spectral Bands"""
        df_smooth = (
            df[spectral_bands]
            .rolling(window=window)
            .mean()
            .add_suffix('_smooth')
        )

        df = pd.concat([df, df_smooth], axis=1)
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.on:
            return df
        df = self.avg_smoothing(df)
        return df