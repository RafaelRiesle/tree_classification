import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils.constants import COLOR, spectral_bands


def plot_intervals_timestamps(df: pd.DataFrame, addition: str = None):
    if addition is None:
        addition = "Overview"

    years = df["year"].unique()
    n_years = len(years)

    fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5), sharey=True)

    if n_years == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        df_year = df[df["year"] == year]
        ax.plot(df_year["time"], df_year["date_diff"], marker="o", linestyle="-")
        ax.set_title(f"Year: {year}")
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Interval (days)")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Intervals between timestamps ({addition})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_top_correlations(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["Columns"], df["Value"], color=COLOR)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("Column Pairs")
    ax.set_ylabel("Correlation")
    ax.set_title("Top-N Correlations")
    ax.set_ylim([-1.5, 1.5])
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_band_differences(df, shift_years=1, date_col="time", n_cols=2):
    df = df.copy()

    if date_col not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{date_col} not found and index is not datetime.")

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    df_shifted = df[spectral_bands].copy()
    df_shifted.index = df_shifted.index + pd.DateOffset(years=shift_years)

    n_bands = len(spectral_bands)
    n_rows = math.ceil(n_bands / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, band in enumerate(spectral_bands):
        ax = axes[i]
        ax.plot(df.index, df[band], label="Original", marker=".", color="blue")
        ax.plot(
            df_shifted.index,
            df_shifted[band],
            label=f"Shifted (+{shift_years}y)",
            marker=".",
            linestyle="--",
            color="red",
        )

        df_common = df[band].reindex(df_shifted.index, method="nearest")
        diff = df_common - df_shifted[band]
        ax.plot(
            df_shifted.index,
            diff,
            label="Diff (Original - Shifted)",
            linestyle=":",
            color="green",
        )

        ax.set_title(f"Band: {band}")
        ax.set_ylabel("Reflectance / Index Value")
        ax.legend()
        ax.grid()

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_autocorrelation_bands(df, bands, lags=26):
    """
    Plot ACF and PACF for multiple spectral bands.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'time' and spectral band columns.
    bands : list of str
        Column names (spectral bands) to analyze.
    lags : int
        Number of lags to show in ACF/PACF.
    """
    df_correlation = df.set_index("time")

    n_bands = len(bands)
    fig, axes = plt.subplots(n_bands, 2, figsize=(14, 2 * n_bands))

    if n_bands == 1:
        axes = [axes]

    for i, band in enumerate(bands):
        y = df_correlation[band].dropna()

        # ACF
        plot_acf(y, lags=lags, alpha=0.05, ax=axes[i][0])
        axes[i][0].set_title(f"Autocorrelation (ACF {band})")

        # PACF
        plot_pacf(y, lags=lags, alpha=0.05, ax=axes[i][1])
        axes[i][1].set_title(f"Partial Autocorrelation (PACF {band})")

    plt.tight_layout()
    plt.show()
