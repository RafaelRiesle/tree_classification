from pathlib import Path
from models.lstm.lstm_utils.data_processor import DataProcessor
from models.lstm.experiments.lstm_trainer import LSTMTrainer
from models.lstm.evaluation.model_evaluator import ModelEvaluator

BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"


def train_model(
    train_path=TRAIN_PATH,
    val_path=TEST_PATH,
    test_path=VAL_PATH,
    batch_size=16,
    lr=0.001,
    max_epochs=5,
):
    # 1. Process data
    processor = DataProcessor(train_path, val_path, test_path, label_column="species")
    data_info = processor.run()

    # 2. Train LSTM
    trainer = LSTMTrainer(
        train_sequences=data_info["train_sequences"],
        val_sequences=data_info["val_sequences"],
        test_sequences=data_info["test_sequences"],
        n_features=len(data_info["feature_columns"]),
        n_classes=data_info["n_classes"],
        class_weights=data_info["class_weights"],
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
    )
    trainer.train()
    model, data_module = trainer.get_model()

    print("Finished Training.")
    return model, data_module, data_info


def evaluate_model(model, data_module, data_info):
    evaluator = ModelEvaluator(
        model, data_module, feature_columns=data_info["feature_columns"]
    )
    evaluator.plot_training_history()
    evaluator.evaluate_confusion_matrix()
    evaluator.permutation_importance()
    print("Finished Evaluation.")


def run():
    model, data_module, data_info = train_model()
    evaluate_model(model, data_module, data_info)
    print("Finished LSTM training and evaluation.")


if __name__ == "__main__":
    run()
