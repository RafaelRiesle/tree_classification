from pathlib import Path
from models.lstm.pipeline.data_processor import DataProcessor 
from models.lstm.experiments.lstm_trainer import LSTMTrainer
from models.lstm.evaluation.model_evaluator import ModelEvaluator

BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
VAL_PATH = BASE_DIR / "data/processed/testset.csv"
TEST_PATH = BASE_DIR / "data/processed/valset.csv"

def run():
    # 1. Process data
    processor = DataProcessor(TRAIN_PATH, VAL_PATH, TEST_PATH, label_column="species")
    data_info = processor.run()

    # 2. Train LSTM
    trainer = LSTMTrainer(
        train_sequences=data_info["train_sequences"],
        val_sequences=data_info["val_sequences"],
        test_sequences=data_info["test_sequences"],
        n_features=len(data_info["feature_columns"]),
        n_classes=data_info["n_classes"],
        class_weights=data_info["class_weights"],
        batch_size=16,
        lr=0.001,
        max_epochs=5,
    )
    trainer.train()
    model, data_module = trainer.get_model()

    # 3. Evaluate model
    evaluator = ModelEvaluator(
        model, data_module, feature_columns=data_info["feature_columns"]
    )
    evaluator.plot_training_history()
    evaluator.evaluate_confusion_matrix()
    evaluator.permutation_importance()
    print("LSTM training and evaluation complete.")


if __name__ == "__main__":
    run()
