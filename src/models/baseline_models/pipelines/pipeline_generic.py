from baseline_utils.baseline_model_manager import BaselineModelManager
from baseline_utils.baseline_pipeline import BasePipeline


class GenericPipeline:
    def __init__(self, target_col="species"):
        self.pipeline = BasePipeline(target_col=target_col)
        self.baseline = BaselineModelManager()

    def run(self, train_df, test_df, model_defs):
        """
        model_defs = [
            (RandomForestClassifier, {"n_estimators": 5, "max_depth": 3}),
            (LogisticRegression, {"max_iter": 200}),
            (SVC, {"kernel": "linear"})
        ]
        """
        X_train, y_train = self.pipeline.fit(train_df)
        X_test, y_test = self.pipeline.transform(test_df)

        for model_class, params in model_defs:
            self.baseline.run_training(
                model_class, params, X_train, y_train, X_test, y_test, X_train.columns
            )

        return self.baseline.load_baseline_models()
