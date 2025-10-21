import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from pathlib import Path
from models.ensemble_models.ensemble_utils.ensemble_model_manager import (
    EnsembleModelManager,
)
from models.ensemble_models.ensemble_utils.ensemble_pipeline import EnsemblePipeline
from sklearn.model_selection import GridSearchCV


class GenericPipeline:
    def __init__(self, target_col="species"):
        self.pipeline = EnsemblePipeline(target_col=target_col)
        self.ensemble = EnsembleModelManager()

    def _print_metrics(self, model_name, metrics, top_n_features=5):
        """
        Nicely prints the model metrics and top feature importances.
        """
        print(f"\n{'=' * 40}\nModel: {model_name}\n{'=' * 40}")

        # Key performance metrics
        metrics_table = [
            ["Accuracy", metrics.get("accuracy")],
            ["Precision (Macro)", metrics.get("precision_macro")],
            ["Recall (Macro)", metrics.get("recall_macro")],
            ["F1 (Macro)", metrics.get("f1_macro")],
        ]
        print("\nPerformance Metrics:")
        print(
            tabulate(
                metrics_table,
                headers=["Metric", "Score"],
                tablefmt="fancy_grid",
                floatfmt=".4f",
            )
        )

        feat_imp = metrics.get("feature_importances")
        if feat_imp:
            feat_imp_df = (
                pd.DataFrame(feat_imp)
                .sort_values(by="importance", ascending=False)
                .head(top_n_features)
            )
            print(f"\nTop {top_n_features} Feature Importances:")
            print(
                tabulate(
                    feat_imp_df, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"
                )
            )

    def run(self, train_df, test_df, model_defs, val_df):
        """
        Trains and evaluates all models defined in model_defs.
        Saves only one JSON per model that contains train, test, and validation metrics.
        """
        X_train, y_train = self.pipeline.fit(train_df)
        X_test, y_test = self.pipeline.transform(test_df)
        X_val, y_val = self.pipeline.transform(val_df)

        feature_names = (
            X_train.columns
            if hasattr(X_train, "columns")
            else [f"f{i}" for i in range(X_train.shape[1])]
        )

        results_summary = []

        for model_class, params in model_defs:
            model_name = model_class.__name__
            print(f"\n{'-' * 30}\nTraining {model_name}...\n{'-' * 30}")

            # GridSearch falls nötig
            if any(isinstance(v, (list, tuple)) for v in params.values()):
                grid = GridSearchCV(model_class(), params, cv=5, n_jobs=-1, scoring="accuracy")
                grid.fit(X_train, y_train)
                hyperparams = grid.best_params_
                print(f"Best hyperparameters: {hyperparams}")
            else:
                hyperparams = params

            # Modell trainieren
            model, _ = self.ensemble.train_and_predict(model_class, hyperparams, X_train, y_train, X_train)

            # Verschiedene Datensätze bewerten
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            val_pred = model.predict(X_val)

            train_metrics = self.ensemble.compute_metrics(y_train, train_pred)
            test_metrics = self.ensemble.compute_metrics(y_test, test_pred)
            val_metrics = self.ensemble.compute_metrics(y_val, val_pred)

            feat_imp_df = self.ensemble.extract_feature_importances(model, feature_names)
            feat_imp = feat_imp_df.to_dict(orient="records") if feat_imp_df is not None else None

            n_samples, n_features = X_train.shape

            # Kombinierte Metriken in einer JSON
            combined_metrics = {
                "train": train_metrics,
                "test": test_metrics,
                "validation": val_metrics,
                "feature_importances": feat_imp,
            }


            model_file = self.ensemble.results_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(model, model_file)

            model_dict = {
                "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "hyperparams": hyperparams,
                "metrics": combined_metrics,
                "features": list(feature_names),
                "n_samples": n_samples,
                "n_features": n_features,
                "model_file": str(model_file),
            }

            # In JSON speichern (nur einmal!)
            self.ensemble.save_to_json(model_dict)

            # Zusammenfassung
            results_summary.append(
                {
                    "model": model_name,
                    "train_acc": train_metrics["accuracy"],
                    "test_acc": test_metrics["accuracy"],
                    "val_acc": val_metrics["accuracy"],
                }
            )

            # Ausgabe im Terminal
            self._print_metrics(model_name, test_metrics)

        return self.ensemble.load_models()
