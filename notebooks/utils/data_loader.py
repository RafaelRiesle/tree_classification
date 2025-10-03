import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    def load_transform(self, path):
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {path} not found.")
        except pd.errors.ParserError:
            raise ValueError(f"File {path} could not be read as CSV.")

        if not {"time", "id"}.issubset(df.columns):
            raise ValueError("CSV must contain 'time' and 'id' columns.")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        if df["time"].isna().any():
            raise ValueError(
                "Some values in 'time' could not be converted to datetime."
            )

        agg_dict = {
            col: "mean" if pd.api.types.is_numeric_dtype(dtype) else "first"
            for col, dtype in df.dtypes.items()
            if col not in ["time", "id"]
        }

        df = df.groupby(["time", "id"], as_index=False).agg(agg_dict)
        df = df.sort_values(["id", "time"]).reset_index(drop=True)
        return df

    def date_feature_extraction(self, df):
        df = df.copy()
        df["month_num"] = df["time"].dt.month
        df["year"] = df["time"].dt.year
        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        df["season"] = df["month_num"].apply(lambda m: seasons[((m % 12) // 3)])
        df["date_diff"] = df.groupby("id")["time"].diff().dt.days
        return df

    def feature_extraction(self, df):
        df = df.copy()
        if "disturbance_year" not in df.columns:
            df["is_disturbed"] = False
        else:
            df["is_disturbed"] = df["disturbance_year"].apply(lambda x: x != 0)
        return df
