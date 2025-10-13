import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    def load_transform(self, path):
        """load data and perform basic aggregation and sorting"""
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
