import pandas as pd


class BasicFeatures:
    def __init__(self):
        pass

    def add_disturbance_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "disturbance_year" not in df.columns:
            df["is_disturbed"] = False
        else:
            df["is_disturbed"] = df["disturbance_year"].apply(lambda x: x != 0)
        return df
