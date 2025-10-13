
from pathlib import Path
from general_utils.utility_functions import load_data
from sklearn.ensemble import RandomForestClassifier
from models.ensemble_models.pipelines.pipeline_generic import GenericPipeline
from sklearn.linear_model import LogisticRegression



BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"

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


def run_ensemble(train_path=TRAIN_PATH, test_path=TEST_PATH, val_path=VAL_PATH):
    train_df, test_df, val_df = load_data(train_path, test_path, val_path)

    models = define_models()
    results = GenericPipeline().run(train_df, test_df, models, val_df=val_df)
    return results


if __name__ == "__main__":
    run_ensemble(train_path=TRAIN_PATH, test_path=TEST_PATH, val_path=VAL_PATH)
