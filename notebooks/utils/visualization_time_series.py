import matplotlib.pyplot as plt

def plot_date_diff_distribution(df, column="date_diff", bins=20, color="black"):
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
