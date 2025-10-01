import pandas as pd


class TimeSeriesAggregate:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["time"] = pd.to_datetime(self.df["time"])

    def aggregate_timeseries(
        self, freq: str = "2W", method: str = "median"
    ) -> pd.DataFrame:
        self.df = self.df.set_index("time")
        num_df = self.df.select_dtypes(include="number")

        agg_methods = {"mean", "median", "sum", "min", "max"}
        if method not in agg_methods:
            raise ValueError(f"Method must be one of {agg_methods}")

        agg_func = getattr(num_df.resample(freq), method)
        agg_df = agg_func().dropna(how="all").reset_index()
        return agg_df
