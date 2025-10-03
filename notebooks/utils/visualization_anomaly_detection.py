import matplotlib.pyplot as plt
import math


def plot_outlier_detection_grid(df, bands, ncols=2):
    n_bands = len(bands)
    nrows = math.ceil(n_bands / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows), sharex=True)
    axes = axes.flatten()

    for i, band in enumerate(bands):
        ax = axes[i]
        ax.plot(df["time"], df[f"{band}_original"], ".-", label="Original", alpha=0.6)
        ax.plot(df["time"], df[band], ".-", label="Cleaned", alpha=0.8)
        outliers = df[df[f"is_outlier_{band}"]]
        ax.scatter(
            outliers["time"],
            outliers[f"{band}_original"],
            color="red",
            s=80,
            marker="x",
            label="Outliers",
        )

        ax.set_title(f"Band: {band}")
        ax.grid(True)
        ax.legend(loc="best")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


def plot_with_outliers_subplot(df, spectral_bands):
    _, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Original Plot ---
    ax = axes[0]
    for band in spectral_bands:
        ax.plot(
            df["time"], df[f"{band}_original"], marker=".", label=f"{band} (original)"
        )
        # Mark outliers
        outliers = df[df["any_outlier"]]
        ax.scatter(
            outliers["time"],
            outliers[f"{band}_original"],
            color="red",
            marker="x",
            s=80,
        )
    ax.set_title("Original with Outliers")
    ax.set_ylabel("Value")
    ax.grid(True)

    # --- Cleaned Plot ---
    ax = axes[1]
    for band in spectral_bands:
        ax.plot(df["time"], df[band], marker=".", label=f"{band} (cleaned)")
    ax.set_title("Cleaned (Interpolated)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    ax.grid(True)

    plt.tight_layout()
    plt.show()
