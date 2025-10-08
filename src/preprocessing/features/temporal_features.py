import pandas as pd
import numpy as np

class TemporalFeatures:
    """
    Extract time-related features
    """

    def __init__(self):
        pass

    def month_and_season(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["month_num"] = df["time"].dt.month
        df["year"] = df["time"].dt.year

        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        df["season"] = df["month_num"].apply(lambda m: seasons[((m % 12) // 3)])

        return df
    
    def month_sin_cos(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["month_num"] = df["time"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        return df

    def date_diff(self, df: pd.DataFrame) -> pd.DataFrame:  
        df = df.copy()
        df["date_diff"] = df.groupby("id")["time"].diff().dt.days
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.month_and_season(df)
        df = self.month_sin_cos
        df = self.date_diff(df)
        return df