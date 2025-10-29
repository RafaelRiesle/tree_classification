import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb

from pipelines.preprocessing.run_preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
from pipelines.processing.processing_pipeline import ProcessingPipeline
from pipelines.processing.features.basic_features import BasicFeatures
from pipelines.processing.features.temporal_features import TemporalFeatures
from pipelines.processing.features.spectral_indices import CalculateIndices
from pipelines.processing.processing_steps.interpolation import Interpolation
from models.baseline_model.baseline_model_utils import (
    drop_unwanted_columns,
    split_into_X_y,
)
from models.baseline_model.calculate_keyfigures import StatisticalFeatures
from general_utils.constants import spectral_bands, indices


class DataReductionAnalysis:
    def __init__(self, base_dir="../../../"):
        self.base_dir = Path(base_dir)
        self.data_path = self.base_dir / "data/raw/raw_trainset.csv"
        self.splits_path = self.base_dir / "data/raw/splits"
        self.preprocessed_path = self.base_dir / "data/preprocessed"
        self.output_dir = self.base_dir / "data/baseline_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features = spectral_bands + indices
        self.df = None

    def run_preprocessing(self):
        run_preprocessing_pipeline(
            data_path=self.data_path,
            splits_output_path=self.splits_path,
            preprocessed_output_path=self.preprocessed_path,
            sample_size=None,
            force_split_creation=True,
        )

        steps = [
            BasicFeatures(on=True),
            Interpolation(on=True),
            CalculateIndices(on=True),
            TemporalFeatures(on=True),
        ]

        pipeline = ProcessingPipeline(
            path=self.preprocessed_path / "trainset.csv", steps=steps
        )
        self.df = pipeline.run()
        self.df["year"] = self.df["time"].dt.year

    def prepare_data(self, df_subset):
        df_subset = drop_unwanted_columns(df_subset)
        df_train = StatisticalFeatures().calculate_keyfigures_per_id(
            df_subset, self.features
        )
        le = LabelEncoder()
        df_train["species_encoded"] = le.fit_transform(df_train["species"])
        X, y, _ = split_into_X_y(df_train, df_train)
        return X, y, le

    def train_xgb(self, X, y, num_classes):
        model = xgb.XGBClassifier(
            n_estimators=2,
            learning_rate=0.1,
            max_depth=10,
            random_state=42,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=num_classes,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def evaluate(self, model, X, y):
        return balanced_accuracy_score(y, model.predict(X))

    def _train_split_plot(self, data_splits, title, xlabel):
        results = []
        for label, df_subset in data_splits:
            X, y, le = self.prepare_data(df_subset)
            split_idx = int(0.8 * len(X))
            model = self.train_xgb(X[:split_idx], y[:split_idx], len(le.classes_))
            results.append(
                {
                    "label": label,
                    "acc": self.evaluate(model, X[split_idx:], y[split_idx:]),
                }
            )

        df_results = pd.DataFrame(results)

        plt.figure(figsize=(8, 5))
        plt.bar(df_results["label"], df_results["acc"], color="royalblue", edgecolor="black")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Balanced Accuracy")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()



    def train_single_years(self):
        data_splits = [
            (year, self.df[self.df["year"] == year])
            for year in sorted(self.df["year"].unique())
        ]
        self._train_split_plot(data_splits, "Balanced Accuracy – Single Years", "Year")

    def train_cumulative_years(self):
        years = sorted(self.df["year"].unique())
        data_splits = [
            ("-".join(map(str, years[:i])), self.df[self.df["year"].isin(years[:i])])
            for i in range(1, len(years) + 1)
        ]
        self._train_split_plot(
            data_splits, "Balanced Accuracy – Cumulative Years", "Years Combined"
        )

    def train_monthly_for_year(self, year):
        df_year = self.df[self.df["year"] == year].copy()
        df_year["month"] = df_year["time"].dt.month
        data_splits = [
            (month, df_year[df_year["month"] == month])
            for month in sorted(df_year["month"].unique())
            if not df_year[df_year["month"] == month].empty
        ]
        self._train_split_plot(
            data_splits, f"Balanced Accuracy per Month ({year})", "Month"
        )

    def train_monthly_per_year(self):
        results = []
        for year in sorted(self.df["year"].unique()):
            df_year = self.df[self.df["year"] == year].copy()
            df_year["month"] = df_year["time"].dt.month
            for month in sorted(df_year["month"].unique()):
                df_month = df_year[df_year["month"] == month]
                if df_month.empty:
                    continue
                X, y, le = self.prepare_data(df_month)
                split_idx = int(0.8 * len(X))
                model = self.train_xgb(X[:split_idx], y[:split_idx], len(le.classes_))
                results.append(
                    {
                        "year": year,
                        "month": month,
                        "acc": self.evaluate(model, X[split_idx:], y[split_idx:]),
                    }
                )
        df_results = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        for year in sorted(df_results["year"].unique()):
            subset = df_results[df_results["year"] == year]
            plt.plot(subset["month"], subset["acc"], marker="o", label=str(year))
        plt.title("Balanced Accuracy per Month and Year")
        plt.xlabel("Month")
        plt.ylabel("Balanced Accuracy")
        plt.legend(title="Year")
        plt.grid(True)
        plt.xticks(range(1, 13))
        plt.tight_layout()
        plt.show()
