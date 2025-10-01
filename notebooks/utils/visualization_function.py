import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import math
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_intervals_timestamps(df: pd.DataFrame, addition: str = None):
    if addition is None:
        addition = "Overview"
    fig = px.line(
        df,
        x="doy",
        y="date_diff",
        title=f"Intervals between timestamps ({addition})",
        facet_col="year",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.show()


def plot_top_correlations(df: pd.DataFrame):
    fig = px.bar(df, x="Columns", y="Value", text_auto=True, title="Top-N Correlations")
    fig.update_layout(yaxis_range=[-1, 1])
    fig.update_layout(xaxis_title="Column Pairs", yaxis_title="Correlation")
    fig.show()


def plot_autocorrelation(df, column):
    df_correlation = df.set_index("time")
    y = df_correlation[column]

    _, axes = plt.subplots(1, 2, figsize=(16, 4))

    # ACF
    plot_acf(y, lags=50, alpha=0.05, ax=axes[0])
    axes[0].set_title(f"Autocorrelation (ACF {column})")

    # PACF
    plot_pacf(y, lags=50, alpha=0.05, ax=axes[1])
    axes[1].set_title(f"partielle autocorrelation partielle (PACF {column})")

    plt.tight_layout()
    plt.show()


def plot_band_differences(df, lag=26, n_cols=2):
    # Identify all the band columns
    band_columns = [col for col in df.columns if col.startswith("b")]

    # Create shifted DataFrame (lag)
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

    # Hide extra subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
