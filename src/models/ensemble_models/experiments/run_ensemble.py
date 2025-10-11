from pathlib import Path
from general_utils.utility_functions import load_data
from sklearn.ensemble import RandomForestClassifier
from models.ensemble_models.pipelines.pipeline_generic import GenericPipeline
from sklearn.linear_model import LogisticRegression


TRAIN_PATH = Path("../../../../data/processed/train.csv")
TEST_PATH = Path("../../../../data/processed/test.csv")


def define_models():
    return [
        (
            RandomForestClassifier,
            {"n_estimators": [5, 10], "max_depth": [1, 2, 4]},
        ),
        (LogisticRegression, {"max_iter": [100, 200]}),
    ]


def train_models(train_df, test_df, models, target_col="species"):
    pipeline = GenericPipeline(target_col=target_col)
    pipeline.run(train_df, test_df, models)
    print("Training finished!")


def run_ensemble(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_df, test_df = load_data(train_path, test_path)
    models = define_models()
    results = train_models(train_df, test_df, models)
    return results


if __name__ == "__main__":
    run_ensemble(train_path=TRAIN_PATH, test_path=TEST_PATH)
