import math
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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
    bars = ax.bar(df["Columns"], df["Value"], color="blue")
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


def plot_autocorrelation(df, column):
    df_correlation = df.set_index("time")
    y = df_correlation[column]

    _, axes = plt.subplots(1, 2, figsize=(16, 4))

    # ACF
    plot_acf(y, lags=20, alpha=0.05, ax=axes[0])
    axes[0].set_title(f"Autocorrelation (ACF {column})")

    # PACF
    plot_pacf(y, lags=20, alpha=0.05, ax=axes[1])
    axes[1].set_title(f"partielle autocorrelation partielle (PACF {column})")

    plt.tight_layout()
    plt.show()


def plot_band_differences(df, lag=52, n_cols=2):
    band_columns = [col for col in df.columns if col.startswith("b")]
    df_shifted = df[band_columns].shift(periods=lag)

    # Calculate differences
    df_diff = df[band_columns] - df_shifted

    # Setup subplots
    n_bands = len(band_columns)
    n_rows = math.ceil(n_bands / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    # Plot each band
    for i, col in enumerate(band_columns):
        axes[i].plot(df.index, df[col], label="Original", marker=".", color="blue")
        axes[i].plot(
            df.index,
            df_shifted[col],
            label="Shifted (t-1Y)",
            linestyle=":",
            color="orange",
        )
        axes[i].plot(
            df.index, df_diff[col], label="Diff (YoY)", linestyle="--", color="green"
        )

        axes[i].set_title(f"Band: {col}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Reflectance / Index Value")
        axes[i].legend()
        axes[i].grid()
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
