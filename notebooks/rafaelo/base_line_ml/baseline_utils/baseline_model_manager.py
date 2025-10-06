import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class BaselineModelManager:
    def __init__(self, results_file="baseline_models.json"):
        self.results_file = results_file
        # Load existing baseline models if the file exists
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                self.baseline_models = json.load(f)
        else:
            self.baseline_models = []

    # ---------------- Train and predict ----------------
    @staticmethod
    def train_and_predict(model_class, hyperparams, X_train, y_train, X_test):
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred

    # ---------------- Compute metrics ----------------
    @staticmethod
    def compute_metrics(y_true, y_pred):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        return metrics

    # ---------------- Extract feature importances ----------------
    @staticmethod
    def extract_feature_importances(model, feature_names):
        """Extract feature importances if the model supports it."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values(by="importance", ascending=False)
            return feat_imp_df
        return None

    # ---------------- Prepare baseline model dictionary ----------------
    @staticmethod
    def prepare_baseline_model_dict(
        run_id, model_class, hyperparams, metrics, feature_names
    ):
        baseline_model = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_class.__name__,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "features": list(feature_names),
        }
        return baseline_model

    # ---------------- Save baseline model to JSON ----------------
    def save_to_json(self, baseline_model):
        baseline_model["run_id"] = len(self.baseline_models) + 1
        self.baseline_models.append(baseline_model)

        with open(self.results_file, "w") as f:
            json.dump(self.baseline_models, f, indent=4)
        print(
            f"Baseline Model #{baseline_model['run_id']} saved to {self.results_file}"
        )

    # ---------------- High-level wrapper to run a baseline model ----------------
    def run_training(
        self, model_class, hyperparams, X_train, y_train, X_test, y_test, feature_names
    ):
        model, y_pred = self.train_and_predict(
            model_class, hyperparams, X_train, y_train, X_test
        )
        metrics = self.compute_metrics(y_test, y_pred)

        # Feature importances
        feat_imp_df = self.extract_feature_importances(model, feature_names)
        metrics["feature_importances"] = (
            feat_imp_df.to_dict(orient="records") if feat_imp_df is not None else None
        )

        baseline_model = self.prepare_baseline_model_dict(
            None, model_class, hyperparams, metrics, feature_names
        )
        self.save_to_json(baseline_model)
        return model, metrics

    # ---------------- Load baseline models as DataFrame ----------------
    def load_baseline_models(self):
        records = []
        for bm in self.baseline_models:
            metrics = bm["metrics"]
            records.append(
                {
                    "run_id": bm.get("run_id"),
                    "timestamp": bm["timestamp"],
                    "model": bm["model"],
                    "accuracy": metrics.get("accuracy"),
                    "precision_macro": metrics.get("precision_macro"),
                    "recall_macro": metrics.get("recall_macro"),
                    "f1_macro": metrics.get("f1_macro"),
                    "params": bm["hyperparams"],
                    "features": bm["features"],
                }
            )
        return pd.DataFrame(records)

    # ---------------- Retrieve a specific baseline model ----------------
    def get_baseline_model_by_id(self, run_id):
        for bm in self.baseline_models:
            if bm["run_id"] == run_id:
                return bm
        raise ValueError(f"Baseline model with run_id={run_id} not found.")

    # ---------------- Plot confusion matrix ----------------
    def plot_confusion_matrix(self, run_id):
        bm = self.get_baseline_model_by_id(run_id)
        cm = bm["metrics"]["confusion_matrix"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {bm['model']} (run_id={run_id})")
        plt.show()

    # ---------------- Print classification report ----------------
    def print_classification_report(self, run_id):
        bm = self.get_baseline_model_by_id(run_id)
        report = bm["metrics"]["classification_report"]

        print(f"\nClassification Report - {bm['model']} (run_id={run_id}):\n")
        for label, scores in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                print(f"Class {label}: {scores}")

    # ---------------- Plot feature importances ----------------
    def plot_feature_importances(self, run_id, top_n=20):
        bm = self.get_baseline_model_by_id(run_id)
        feat_imp = bm["metrics"].get("feature_importances")
        if feat_imp is None:
            print(f"No feature importances for run_id={run_id}")
            return

        feat_imp_df = (
            pd.DataFrame(feat_imp)
            .sort_values(by="importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=feat_imp_df)
        plt.title(f"Top {top_n} Feature Importances - {bm['model']} (run_id={run_id})")
        plt.show()

    # ---------------- Plot performance distribution  ----------------
    def plot_performance_distribution(self, column):
        """
        Plot accuracy of all baseline models over time, grouped by model type.
        """
        df_results = self.load_baseline_models()
        df_results["timestamp"] = pd.to_datetime(df_results["timestamp"])

        plt.figure(figsize=(10, 6))
        for model in df_results["model"].unique():
            subset = df_results[df_results["model"] == model]
            plt.plot(subset["timestamp"], subset[column], marker="o", label=model)

        plt.ylabel("Accuracy")
        plt.xlabel("Timestamp")
        plt.title("Baseline Model Accuracies Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
