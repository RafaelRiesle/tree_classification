import pandas as pd
import numpy as np


class CSVDataLoader:
    def load_transform(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if not {"time", "id"}.issubset(df.columns):
            raise ValueError("CSV must contain 'time' and 'id' columns.")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        if df["time"].isna().any():
            raise ValueError("Some 'time' values could not be converted to datetime.")
        agg = {
            col: "mean" if pd.api.types.is_numeric_dtype(dtype) else "first"
            for col, dtype in df.dtypes.items()
            if col not in ["time", "id"]
        }
        return (
            df.groupby(["time", "id"], as_index=False)
            .agg(agg)
            .sort_values(["id", "time"])
            .drop_duplicates()
            .reset_index(drop=True)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
