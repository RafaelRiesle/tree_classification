from pathlib import Path
from pipelines.preprocessing.run_preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
from models.ensemble_models.experiments.run_ensemble import run_ensemble
from models.ensemble_models.evaluation.evaluate_models import run_ensemble_evaluation
from models.lstm.experiments.run_lstm import run_lstm
from models.lstm.validation.evaluate import run_lstm_evaluation


# === PATH CONFIGURATION ===
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

paths = {
    "train_path": PROCESSED_DIR / "trainset.csv",
    "test_path": PROCESSED_DIR / "testset.csv",
    "val_path": PROCESSED_DIR / "valset.csv",
}


# === MAIN PIPELINE ===
def run_training_pipeline():
    print("=== Starting Training Pipeline ===")

    # [1] Preprocessing
    print("[1] Running preprocessing...")
    run_preprocessing_pipeline(
        data_path=RAW_DIR / "raw_trainset.csv",
        splits_output_path=RAW_DIR / "splits",
        preprocessed_output_path=PREPROCESSED_DIR,
        sample_size=200,
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1,
        remove_outliers=True,
        contamination=0.05,
        force_split_creation=False,
    )
    print("[1] Preprocessing complete.\n")

    # [2] Additional processing (placeholder)
    print("[2] Running additional processing...")
    # TODO: add extra preprocessing or feature engineering here
    print("[2] Additional processing complete.\n")

    # [3] Ensemble model training
    print("[3] Training ensemble models...")
    run_ensemble(**paths)
    run_ensemble_evaluation()
    print("[3] Ensemble models complete.\n")

    # [4] LSTM model training
    print("[4] Training LSTM model...")
    run_lstm(**paths, batch_size=16, lr=1e-3, max_epochs=2)
    run_lstm_evaluation(**paths, batch_size=50)
    print("[4] LSTM training complete.\n")

    print("=== Training Pipeline Finished ===")


if __name__ == "__main__":
    run_training_pipeline()
