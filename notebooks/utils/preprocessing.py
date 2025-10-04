import pandas as pd
import numpy as np

class Preprocessing:
    
    @staticmethod
    def interpolate_b4(df: pd.DataFrame, method="linear") -> pd.DataFrame:
        """interpolate b4 = 0 values"""
        df.loc[df["b4"] == 0, "b4"] = np.nan
        df = df.sort_values(["id", "time"])
        df["b4"] = df.groupby("id")["b4"].transform(
            lambda x: x.interpolate(method=method)
        )
        return df
