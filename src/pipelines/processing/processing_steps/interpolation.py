import pandas as pd
import numpy as np


class Interpolation:
    def __init__(self, on=True):
        self.on = on

    def interpolate_b4(self, df: pd.DataFrame, method="linear") -> pd.DataFrame:
        """interpolate b4 = 0 values"""
        df.loc[df["b4"] == 0, "b4"] = np.nan
        df = df.sort_values(["id", "time"])
        df["b4"] = df.groupby("id")["b4"].transform(
            lambda x: x.interpolate(method=method)
        )
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.on:
            return df
        df = self.interpolate_b4(df)
        return df
