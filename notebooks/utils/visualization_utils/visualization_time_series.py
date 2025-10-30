from utils.constants import COLOR
import matplotlib.pyplot as plt
import pandas as pd
import math


def plot_date_diff_distribution(df, column="date_diff", bins=20, color=COLOR):
    data = df[column].dropna()
    median_val = data.median()

    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, color=color)

    plt.axvline(
        median_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.1f} days",
    )

    plt.title("Distribution of Time Gaps")
    plt.xlabel("Days Between Observations")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_timeseries_oneid(
    df, id, col, df_yearly=None, yearly_mean=True, df_scaled=True
):
    df_sub = df[df["id"] == id]

    plt.plot(df_sub["time"], df_sub[col], label="all data")

    if yearly_mean:
        df_yearly_sub = df_yearly[df_yearly["id"] == id].copy()
        df_yearly_sub["year_dt"] = pd.to_datetime(
            df_yearly_sub["year"].astype(str) + "-01-01"
        )
        plt.plot(df_yearly_sub["year_dt"], df_yearly_sub[col], label="Yearly mean")

    if df_scaled:
        plt.title(f"{col} (normalized) Timeseries for ID {id}")
    else:
        plt.title(f"{col} Timeseries for ID {id}")

    plt.xlabel("Time")
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_timeseries_multiple_ids(
    df, ids, col, df_yearly=None, yearly_mean=True, df_scaled=True
):
    n = len(ids)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, nrows * 4), sharex=False
    )
    axes = axes.flatten()  # um immer eine 1D-Liste zu haben

    for i, id in enumerate(ids):
        ax = axes[i]
        df_sub = df[df["id"] == id]
        ax.plot(df_sub["time"], df_sub[col], label="all data")

        if yearly_mean and df_yearly is not None:
            df_yearly_sub = df_yearly[df_yearly["id"] == id].copy()
            df_yearly_sub["year_dt"] = pd.to_datetime(
                df_yearly_sub["year"].astype(str) + "-01-01"
            )
            ax.plot(df_yearly_sub["year_dt"], df_yearly_sub[col], label="Yearly mean")

        title = (
            f"{col} (normalized) Timeseries for ID {id}"
            if df_scaled
            else f"{col} Timeseries for ID {id}"
        )
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend()

    # leere Subplots entfernen
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
