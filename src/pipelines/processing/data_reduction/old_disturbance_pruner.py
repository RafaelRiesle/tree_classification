import pandas as pd


class OldDisturbancePruner:
    def __init__(self, on=True):
        self.on = on

    def delete_old_disturbances(self, df: pd.DataFrame):
        if "disturbance_year" not in df.columns or "species" not in df.columns:
            return df
        mask = ~((df["disturbance_year"] < 2017) & (df["species"] == "disturbed"))
        return df.loc[mask].copy()

    def run(self, df):
        if self.on:
            return self.delete_old_disturbances(df)
        return df
