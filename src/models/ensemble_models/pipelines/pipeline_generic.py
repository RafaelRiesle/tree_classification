import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
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
        Optionally evaluates a validation set.
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

            if any(isinstance(v, (list, tuple)) for v in params.values()):
                grid = GridSearchCV(
                    model_class(), params, cv=5, n_jobs=-1, scoring="accuracy"
                )
                grid.fit(X_train, y_train)
                hyperparams = grid.best_params_
                print(f"Best hyperparameters: {hyperparams}")
            else:
                hyperparams = params

            model, train_metrics = self.ensemble.run_training(
                model_class,
                hyperparams,
                X_train,
                y_train,
                X_train,
                y_train,
                feature_names,
            )

            _, test_metrics = self.ensemble.run_training(
                model_class,
                hyperparams,
                X_train,
                y_train,
                X_test,
                y_test,
                feature_names,
            )

            _, val_metrics = self.ensemble.run_training(
                model_class,
                hyperparams,
                X_train,
                y_train,
                X_val,
                y_val,
                feature_names,
            )

            results_summary.append(
                {
                    "model": model_name,
                    "train_acc": train_metrics["accuracy"],
                    "test_acc": test_metrics["accuracy"],
                    "val_acc": val_metrics["accuracy"]
                }
            )

            self._print_metrics(model_name, test_metrics)
        return self.ensemble.load_models()

    def _plot_performance_comparison(self, results_summary):
        df = pd.DataFrame(results_summary)
        df.set_index("model", inplace=True)
        df.plot(kind="bar", figsize=(10, 6))
        plt.ylabel("Accuracy")
        plt.title("Train / Test / Validation Performance Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()
