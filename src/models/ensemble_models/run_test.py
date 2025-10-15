import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib


# -----------------------------
# 🧱 Time Series Aggregation
# -----------------------------
class TimeSeriesAggregate:
    def __init__(self, df: pd.DataFrame, static_numeric_cols: list[str] = None):
        self.df = df.copy()
        self.df["time"] = pd.to_datetime(self.df["time"])
        self.static_numeric_cols = static_numeric_cols if static_numeric_cols else []

    def aggregate_timeseries(
        self, freq: str = "2W", method: str = "mean"
    ) -> pd.DataFrame:
        result = []

        for id_val, group in self.df.groupby("id"):
            group = group.set_index("time").sort_index()

            all_num = group.select_dtypes(include="number")
            obj_df = group.select_dtypes(exclude="number")

            static_num_df = all_num[self.static_numeric_cols]
            dynamic_num_df = all_num.drop(columns=self.static_numeric_cols)

            if method not in {"mean", "median", "sum", "min", "max"}:
                raise ValueError(f"Invalid method: {method}")

            agg_dynamic = getattr(dynamic_num_df.resample(freq), method)()
            agg_static = static_num_df.resample(freq).first()
            agg_obj = obj_df.resample(freq).first()

            agg_group = pd.concat([agg_dynamic, agg_static, agg_obj], axis=1)
            agg_group["id"] = id_val

            # Optional: Entferne leere Zeitfenster
            agg_group = agg_group.dropna(how="all", subset=dynamic_num_df.columns)

            result.append(agg_group.reset_index())

        return pd.concat(result, ignore_index=True)


# -----------------------------
# 📥 Load and Aggregate
# -----------------------------
def load_and_aggregate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["id", "time"])
    print(f"basis_df {df.shape}")
    agg = TimeSeriesAggregate(df, static_numeric_cols=["id"])
    df_agg = agg.aggregate_timeseries(freq="3W", method="mean")

    drop_cols = ["doy", "disturbance_year"]
    df_agg = df_agg.drop(columns=[col for col in drop_cols if col in df_agg.columns])

    return df_agg


# -----------------------------
# 🔍 Feature Extraction
# -----------------------------
def extract_tsfresh_features(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    relevant_cols = ["id", "time"] + [
        col for col in numerical_cols if col not in ["id", "time"]
    ]
    df_num = df[relevant_cols]

    features = extract_features(
        df_num,
        column_id="id",
        column_sort="time",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=False,
    )

    features.index.name = "id"
    return features.reset_index()


# -----------------------------
# 🧠 Feature Selection (RandomForest)
# -----------------------------
def rf_feature_ranking(X: pd.DataFrame, y: pd.Series, top_n: int = 70):
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    top_features = importance_df.head(top_n)["feature"].tolist()
    return X[top_features], top_features


# -----------------------------
# 🏁 Model Training (RandomForest)
# -----------------------------
def train_random_forest_model(X, y, label_encoder):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Evaluation on Test Set:")
    print(
        classification_report(
            label_encoder.inverse_transform(y_test),
            label_encoder.inverse_transform(y_pred),
        )
    )
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save model
    joblib.dump(model, "random_forest_model.pkl")
    print("💾 Modell gespeichert: random_forest_model.pkl")

    return model


# -----------------------------
# 🔁 Hauptausführung
# -----------------------------
if __name__ == "__main__":
    # 1. Load & aggregate data
    path = "/Users/rafaelriesle/Documents/Rafael Riesle/Studium/Semster 7/AWP2/tree_classification/data/preprocessed/testset.csv"
    df = load_and_aggregate(path)
    print(df.shape)
    print(df.head())

    if "species" not in df.columns:
        raise ValueError(
            "❌ 'species'-Spalte fehlt. Stelle sicher, dass du den Trainingsdatensatz nutzt."
        )

    # 2. Feature extraction
    features = extract_tsfresh_features(df)

    # 3. Merge Labels
    labels = df[["id", "species"]].drop_duplicates()
    final_df = features.merge(labels, on="id", how="left")

    # 4. Entferne seltene Klassen (Stratify-Split braucht ≥2 Samples)
    label_counts = final_df["species"].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    final_df = final_df[final_df["species"].isin(valid_labels)]

    # 5. Label-Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(final_df["species"])

    # 6. Feature-Matrix
    X = final_df.drop(columns=["id", "species"])
    X = X.fillna(0)

    # 7. Remove constant features
    vt = VarianceThreshold(threshold=0.0)
    X_var = vt.fit_transform(X)
    X_cols = X.columns[vt.get_support()]
    X = pd.DataFrame(X_var, columns=X_cols)
    print(X)

    # 8. Modellbasierte Feature Selektion mit RandomForest
    X_selected, top_features = rf_feature_ranking(X, y, top_n=50)
    print(f"🏆 Top {len(top_features)} Features ausgewählt.")

    # 9. Modelltraining mit RandomForest
    model = train_random_forest_model(X_selected, y, label_encoder)
