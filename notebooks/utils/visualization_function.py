import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
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
    fig.update_layout(yaxis_range=[-1,1])
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