from pathlib import Path
from initial_pipeline.run_initial_pipeline import run_initial_pipeline
from models.ensemble_models.experiments.run_ensemble import run_ensemble
from models.ensemble_models.evaluation.evaluate_models import run_evaluation
from models.lstm.experiments.run_lstm import run_lstm
from models.lstm.validation.evaluate import run_lstm_evaluation
from pipelines.processing.processing_pipeline import ProcessingPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_trainset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "splits"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_OUTPUT_PATH / "train.csv"
TEST_PATH = PROCESSED_OUTPUT_PATH / "test.csv"
VAL_PATH = PROCESSED_OUTPUT_PATH / "val.csv"

def run_training_pipeline():
    print("=== Running training pipeline ===")
    
    print("[1] Running initial pipeline (preprocessing)")
    run_initial_pipeline(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        processed_output=PROCESSED_OUTPUT_PATH,
        contamination=0.05,
    )
    print("[1] Preprocessing finished")

    print("[2] Placeholder for additional preprocessing")

    pipeline = ProcessingPipeline(
        df=df,# lieber path, 
        basic_features=True,
        data_augmentation=False,
        calculate_indices=True,
        temporal_features=True,
        interpolate_b4=True,
        outlier_cleaner=False,
        detect_disturbed_trees=False, 
        specify_disturbed_labels=True,
    )

    df_cleaned = pipeline.run()
    print("[2] Additional preprocessing finished")

    print("[3] Training ensemble models")
    run_ensemble(train_path=TRAIN_PATH, test_path=TEST_PATH)
    run_evaluation()
    print("[3] Ensemble training finished")

    print("[4] Training LSTM model")
    run_lstm(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        val_path=VAL_PATH,
        batch_size=16,
        lr=1e-3,
        max_epochs=2,
    )
    run_lstm_evaluation(batch_size=50)
    print("[4] LSTM training finished")

if __name__ == "__main__":
    run_training_pipeline()

