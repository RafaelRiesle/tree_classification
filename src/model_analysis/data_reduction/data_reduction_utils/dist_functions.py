import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score
from general_utils.constants import spectral_bands


def plot_distribution(df):
    fig = px.histogram(
        df,
        x="relative_year",
        title="Distribution of Disturbed Trees Over Relative Years",
    )
    return fig


def evaluate_time_windows(df):
    df["species_disturbed"] = (df["species"] == "disturbed").astype(int)
    df["year"] = df["time"].dt.year
    df["relative_year"] = df["year"] - df["disturbance_year"]

    results = []

    for window in range(0, 5):
        subset = df[(df["relative_year"] >= -window) & (df["relative_year"] <= window)]
        X, y, groups = subset[spectral_bands], subset["species_disturbed"], subset["id"]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        model = RandomForestClassifier(
            random_state=42, class_weight="balanced", n_estimators=30
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        acc = balanced_accuracy_score(y.iloc[test_idx], preds)
        results.append(
            {"years_around_event": f"{-window}..{window}", "balanced_accuracy": acc}
        )

    results_df = pd.DataFrame(results)
    fig = px.line(
        results_df,
        x="years_around_event",
        y="balanced_accuracy",
        markers=True,
        title="Balanced Accuracy for Different Time Windows Around Disturbance",
        labels={
            "years_around_event": "Time Window (Years)",
            "balanced_accuracy": "Balanced Accuracy",
        },
    )
    fig.show()
