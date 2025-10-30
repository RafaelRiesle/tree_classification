import json
import joblib
from pathlib import Path
import subprocess
from datetime import datetime
import getpass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class EnsembleModelManager:
    def __init__(self, results_dir=None):
        if results_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            results_dir = (
                project_root / "data" / "ensemble_training" / "ensemble_results"
            )

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.models = self.aggregate_results()

    # ---------------- Train & Predict ----------------
    @staticmethod
    def train_and_predict(model_class, hyperparams, X_train, y_train, X_test):
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred

    # ---------------- Compute Metrics ----------------
    @staticmethod
    def compute_metrics(y_true, y_pred):
        return {
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

    # ---------------- Feature Importances ----------------
    @staticmethod
    def extract_feature_importances(model, feature_names):
        if hasattr(model, "feature_importances_"):
            return pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values(by="importance", ascending=False)
        return None

    # ---------------- Prepare Baseline Dict ----------------
    @staticmethod
    def prepare_model_dict(model_class, hyperparams, metrics, feature_names):
        return {
            "run_id": None,
            "timestamp": datetime.now().isoformat(),
            "model": model_class.__name__,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "features": list(feature_names),
        }

    # ---------------- Save Baseline Model ----------------
    def save_to_json(self, model):
        try:
            git_user = subprocess.check_output(
                ["git", "config", "user.name"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            git_user = getpass.getuser()

        user_dir = self.results_dir / git_user
        user_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_file = user_dir / f"run_{timestamp}.json"
        model["run_id"] = timestamp

        with open(run_file, "w") as f:
            json.dump(model, f, indent=4)

        print(f"Baseline Model saved to {run_file}")
        self.models.append(model)

    # ---------------- Aggregate all results ----------------
    def aggregate_results(self):
        all_results = []
        if not self.results_dir.exists():
            return all_results
        for user_dir in self.results_dir.iterdir():
            if user_dir.is_dir():
                for f in sorted(user_dir.glob("*.json")):
                    with open(f, "r") as file:
                        all_results.append(json.load(file))
        return all_results

    # ---------------- Run Training Wrapper ----------------
    def run_training(
        self, model_class, hyperparams, X_train, y_train, X_test, y_test, feature_names
    ):
        model, y_pred = self.train_and_predict(
            model_class, hyperparams, X_train, y_train, X_test
        )
        metrics = self.compute_metrics(y_test, y_pred)

        feat_imp_df = self.extract_feature_importances(model, feature_names)
        metrics["feature_importances"] = (
            feat_imp_df.to_dict(orient="records") if feat_imp_df is not None else None
        )

        model_dict = self.prepare_model_dict(
            model_class, hyperparams, metrics, feature_names
        )

        model_file = (
            self.results_dir
            / f"{model_dict['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        )
        joblib.dump(model, model_file)
        model_dict["model_file"] = str(model_file)

        self.save_to_json(model_dict)
        return model_dict, metrics

    # ---------------- Load trained model by run_id ----------------
    def load_model_by_id(self, run_id):
        bm = self.get_model_by_id(run_id)
        if "model_file" not in bm:
            raise ValueError(f"No trained model saved for run_id={run_id}")
        return joblib.load(bm["model_file"])

    # ---------------- Get best model ----------------
    def get_best_model(self, metric="accuracy", ascending=False):
        df_results = self.load_models()
        if metric not in df_results.columns:
            raise ValueError(
                f"Metric '{metric}' not found. Available: {df_results.columns.tolist()}"
            )

        df_sorted = df_results.sort_values(metric, ascending=ascending).reset_index(
            drop=True
        )
        best_row = df_sorted.iloc[0]
        best_run_id = best_row["run_id"]
        best_model = self.get_model_by_id(best_run_id)
        best_model["trained_model"] = self.load_model_by_id(best_run_id)
        return best_run_id, best_model, df_sorted

    # ---------------- Save best model ----------------
    def save_best_model(self, metric="accuracy", save_dir=None):
        """
        Save the best trained model according to a given metric.
        """
        if save_dir is None:
            save_dir = self.results_dir / "best_model"
        save_dir.mkdir(parents=True, exist_ok=True)

        best_run_id, best_model_dict, _ = self.get_best_model(metric=metric)
        trained_model = best_model_dict["trained_model"]

        model_file = save_dir / f"best_model_{metric}_{best_run_id}.joblib"
        joblib.dump(trained_model, model_file)

        print(f"Best model saved: {model_file}")
        return model_file

    # ---------------- Load models as DataFrame ----------------
    def load_models(self):
        records = []
        for bm in self.models:
            metrics = bm.get("metrics", {})
            records.append(
                {
                    "run_id": bm.get("run_id"),
                    "timestamp": bm.get("timestamp"),
                    "model": bm.get("model"),
                    "accuracy": metrics.get("accuracy"),
                    "precision_macro": metrics.get("precision_macro"),
                    "recall_macro": metrics.get("recall_macro"),
                    "f1_macro": metrics.get("f1_macro"),
                    "params": bm.get("hyperparams"),
                    "features": bm.get("features"),
                }
            )
        return pd.DataFrame(records)

    # ---------------- Retrieve a specific model ----------------
    def get_model_by_id(self, run_id):
        for bm in self.models:
            if bm["run_id"] == run_id:
                return bm
        raise ValueError(f"Baseline model with run_id={run_id} not found.")

    # ---------------- Plot confusion matrix ----------------
    def plot_confusion_matrix(self, run_id):
        bm = self.get_model_by_id(run_id)
        cm = bm["metrics"]["confusion_matrix"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {bm['model']} (run_id={run_id})")
        plt.show()

    # ---------------- Print classification report ----------------
    def print_classification_report(self, run_id):
        bm = self.get_model_by_id(run_id)
        report = bm["metrics"]["classification_report"]

        print(f"\nClassification Report - {bm['model']} (run_id={run_id}):\n")
        for label, scores in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                print(f"Class {label}: {scores}")

    # ---------------- Plot feature importances ----------------
    def plot_feature_importances(self, run_id, top_n=20):
        bm = self.get_model_by_id(run_id)
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

    # ---------------- Plot performance distribution ----------------
    def plot_performance_distribution(self, column):
        df_results = self.load_models()
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
