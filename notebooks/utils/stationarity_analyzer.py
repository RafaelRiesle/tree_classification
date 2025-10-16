import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings

warnings.filterwarnings("ignore", category=InterpolationWarning)


class StationarityAnalyzer:
    def __init__(self, df_base, species="oak", year=2020):
        self.df_base = df_base
        self.species = species
        self.year = year
        self.df_trees = self._prepare_subset()

    def _prepare_subset(self):
        """Filter and prepare the dataset for the specified species and year."""
        df = self.df_base.drop(
            columns=["doy", "disturbance_year", "id"], errors="ignore"
        )
        df["time"] = pd.to_datetime(self.df_base["time"])

        df_filtered = (
            df[(df["species"] == self.species) & (df["time"].dt.year == self.year)]
            .groupby(["time", "species"])
            .mean()
            .reset_index()
        )
        return df_filtered

    @staticmethod
    def adf_test(series):
        """Perform the Augmented Dickey-Fuller (ADF) test for stationarity."""
        result = adfuller(series, autolag="AIC")
        print("ADF Test:")
        print(f"  Test Statistic: {result[0]:.4f}")
        print(f"  p-value:        {result[1]:.4f}")
        if result[1] < 0.05:
            print("  ⇒ Reject null hypothesis: Stationary\n")
        else:
            print("  ⇒ Fail to reject null hypothesis: Not stationary\n")

    @staticmethod
    def kpss_test(series):
        """Perform the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test for stationarity."""
        result = kpss(series, regression="c", nlags="auto")
        print("KPSS Test:")
        print(f"  Test Statistic: {result[0]:.4f}")
        print(f"  p-value:        {result[1]:.4f}")
        if result[1] < 0.05:
            print("  ⇒ Reject null hypothesis: Not stationary\n")
        else:
            print("  ⇒ Fail to reject null hypothesis: Stationary\n")

    def apply_stl_and_tests(self, bands):
        for band in bands:
            if band not in self.df_trees.columns:
                print(f"⚠️ Skipping {band}: column not found.\n")
                continue

            print(f"\n=== {band.upper()} ===")
            self.adf_test(self.df_trees[band])
            self.kpss_test(self.df_trees[band])

            stl = STL(self.df_trees[band], period=12, robust=True)
            res = stl.fit()

            self.df_trees[f"{band}_adjusted"] = res.resid

    @staticmethod
    def plot_stl_per_band(
        df,
        spectral_bands,
        date_col="time",
        n_cols=2,
        period=12,
        species="oak",
        year=2020,
    ):
        """
        Perform STL decomposition for all spectral bands and plot Original, Trend, Seasonal, and Residual.
        """

        df = df.copy()

        # Ensure datetime index
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{date_col} not found and index is not datetime.")

        # Filter by species and year
        df_filtered = df[(df.index.year == year) & (df["species"] == species)]

        # Aggregate daily mean
        df_filtered = (
            df_filtered.groupby([df_filtered.index, "species"]).mean().reset_index()
        )
        df_filtered.set_index(date_col, inplace=True)

        n_bands = len(spectral_bands)
        n_rows = math.ceil(n_bands / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
        axes = axes.flatten()

        for i, band in enumerate(spectral_bands):
            ax = axes[i]
            if band not in df_filtered.columns:
                ax.set_visible(False)
                continue

            # STL decomposition
            stl = STL(df_filtered[band], period=period, robust=True)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            residual = res.resid

            # Store residuals as adjusted column
            df_filtered[f"{band}_adjusted"] = residual

            # Plot original, trend, seasonal, residual
            ax.plot(
                df_filtered.index,
                df_filtered[band],
                label="Original",
                color="gray",
                alpha=0.5,
            )
            ax.plot(df_filtered.index, trend, label="Trend", color="orange")
            ax.plot(df_filtered.index, seasonal, label="Seasonal", color="green")
            ax.plot(df_filtered.index, residual, label="Residual", color="royalblue")

            ax.set_title(f"STL Decomposition: {band.upper()}", fontsize=10)
            ax.set_ylabel("Reflectance / Index Value")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8, loc="upper right")

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Remove unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(
            f"STL Decomposition per Spectral Band ({species}, {year})",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()
