import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
from general_utils.constants import spectral_bands, indices

class TimeSeriesAggregator:
    DEFAULT_FEATURES = spectral_bands + indices

    def __init__(self, features=None, window=120, step=90, n_jobs=-1):
        self.features = features or self.DEFAULT_FEATURES
        self.window = window
        self.step = step
        self.n_jobs = n_jobs

    @staticmethod
    def calc_trend(x, y):
        if len(x) < 2:
            return 0
        xm, ym = np.mean(x), np.mean(y)
        denom = np.sum((x - xm) ** 2)
        return 0 if denom == 0 else np.sum((x - xm) * (y - ym)) / denom

    def process_id(self, id_, group):
        group = group.sort_values("time")
        species = group["species"].iloc[0]
        times = group["time_num"].values
        feats = []
        start = 0

        while start < times.max():
            end = start + self.window
            win = group[(group["time_num"] >= start) & (group["time_num"] < end)]
            if len(win) < 2:
                start += self.step
                continue

            f = {"id": id_, "species": species}
            for col in self.features:
                vals = win[col].values
                t = win["time_num"].values
                f[f"{col}_mean"] = vals.mean()
                f[f"{col}_std"] = vals.std()
                f[f"{col}_min"] = vals.min()
                f[f"{col}_max"] = vals.max()
                f[f"{col}_trend"] = self.calc_trend(t, vals)
            feats.append(f)
            start += self.step
        return feats

    def aggregate_time_series(self, df):
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["time_num"] = (df["time"] - df["time"].min()).dt.days
        groups = list(df.groupby("id"))

        with tqdm_joblib(tqdm(desc="Processing IDs", total=len(groups))) as progress_bar:
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_id)(i, g) for i, g in groups
            )

        return pd.DataFrame([f for sub in res for f in sub])

    def aggregate_to_single_row_keep_windows(self, df_windows):
        df_windows = df_windows.copy()
        df_windows["window_idx"] = df_windows.groupby("id").cumcount()
        df_pivot = df_windows.pivot_table(index=["id","species"], columns="window_idx")
        df_pivot.columns = [f"{col[0]}_w{col[1]}" if isinstance(col, tuple) else col for col in df_pivot.columns]
        return df_pivot.reset_index()

    def run(self, df):
        """Shortcut: vollständiger Lauf."""
        df_win = self.aggregate_time_series(df)
        return self.aggregate_to_single_row_keep_windows(df_win)
