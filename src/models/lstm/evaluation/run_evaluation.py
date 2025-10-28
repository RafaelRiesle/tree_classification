from pathlib import Path
import torch
from models.lstm.lstm_utils.data_processor import DataProcessor
from models.lstm.lstm_utils.data_module import SpeciesDataModule
from models.lstm.lstm_utils.species_predictor import SpeciesPredictor
from models.lstm.evaluation.model_evaluator import ModelEvaluator

BASE_DIR = Path(__file__).parents[4]

# 🔹 Passe den Pfad zu deinem Checkpoint an
CHECKPOINT_PATH = BASE_DIR / "data/lstm_training/results/species_model-epoch=33-val_loss=0.2559.ckpt"

TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"


def evaluate_only(
    ckpt_path=CHECKPOINT_PATH,
    train_path=TRAIN_PATH,
    val_path=TEST_PATH,
    test_path=VAL_PATH,
    batch_size=32,
    lr=1e-3,
):
    # 1️⃣ Daten vorbereiten (wie beim Training)
    processor = DataProcessor(train_path, val_path, test_path, label_column="species")
    data_info = processor.run()

    # 2️⃣ Modell aus Checkpoint laden (SpeciesPredictor, NICHT LSTMTrainer!)
    print(f"Lade Modell aus Checkpoint: {ckpt_path}")
    model = SpeciesPredictor.load_from_checkpoint(
        ckpt_path,
        n_features=len(data_info["feature_columns"]),
        n_classes=data_info["n_classes"],
        lr=lr,
        class_weights=data_info["class_weights"],
    )
    model.eval()

    # 3️⃣ DataModule rekonstruieren
    data_module = SpeciesDataModule(
        data_info["train_sequences"],
        data_info["val_sequences"],
        data_info["test_sequences"],
        batch_size=batch_size,
    )

    # 4️⃣ Evaluation
    evaluator = ModelEvaluator(
        model,
        data_module,
        feature_columns=data_info["feature_columns"],
    )

    evaluator.evaluate_confusion_matrix()
    evaluator.evaluate_metrics() 
    evaluator.permutation_importance()
    print("✅ Finished Evaluation.")


if __name__ == "__main__":
    evaluate_only()
