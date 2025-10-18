import pandas as pd


class BasicFeatures:
    def __init__(self, on=True):
        self.on = on

    def add_disturbance_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "disturbance_year" not in df.columns:
            df["is_disturbed"] = False
        else:
            df["is_disturbed"] = df["disturbance_year"].apply(lambda x: x != 0)
        return df

    def delete_doy(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "doy" in df.columns:
            df = df.drop(columns=["doy"])
        return df

    def run(self, df):
        if not self.on:
            return df
        df = self.add_disturbance_flag(df)
        df = self.delete_doy(df)
        return df
