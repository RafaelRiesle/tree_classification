from pathlib import Path
from models.lstm.lstm_utils.data_processor import DataProcessor
from models.lstm.lstm_utils.data_module import SpeciesDataModule
from models.lstm.lstm_utils.species_predictor import SpeciesPredictor
from models.lstm.evaluation.model_evaluator import ModelEvaluator


BASE_DIR = Path(__file__).parents[4]

CHECKPOINT_PATH = (
    BASE_DIR / "data/lstm_training/results/species_model-epoch=33-val_loss=0.2559.ckpt"
)
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"


def evaluate_only(ckpt_path=CHECKPOINT_PATH, batch_size=32, lr=1e-3):
    processor = DataProcessor(TRAIN_PATH, TEST_PATH, VAL_PATH, label_column="species")
    data_info = processor.run()
    model = SpeciesPredictor.load_from_checkpoint(
        ckpt_path,
        n_features=len(data_info["feature_columns"]),
        n_classes=data_info["n_classes"],
        lr=lr,
        class_weights=data_info["class_weights"],
    )
    model.eval()
    data_module = SpeciesDataModule(
        data_info["train_sequences"],
        data_info["val_sequences"],
        data_info["test_sequences"],
        batch_size=batch_size,
    )

    evaluator = ModelEvaluator(
        model, data_module, feature_columns=data_info["feature_columns"]
    )

    for split in ["test", "val"]:
        print(f"\n===== {split.upper()} METRICS =====")
        evaluator.evaluate_confusion_matrix(split=split)
        evaluator.evaluate_metrics(split=split)
        evaluator.integrated_gradients_analysis(
            target_classes=list(range(data_info["n_classes"])),
            n_samples=1,
            split=split,
            save_plots=True,
        )
        # evaluator.permutation_importance(split=split)

    print("\nFinished Evaluation.")


if __name__ == "__main__":
    evaluate_only()
