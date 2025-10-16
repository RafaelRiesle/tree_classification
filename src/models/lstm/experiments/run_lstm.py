from pathlib import Path
import pytorch_lightning as pl
from models.lstm.lstm_utils.species_data_module import SpeciesDataModule
from models.lstm.lstm_utils.species_predictor import SpeciesPredictor
from models.lstm.pipeline.pipeline import prepare_data

BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"


def create_data_module(data, batch_size: int):
    return SpeciesDataModule(
        train_seqs=data["train_sequences"],
        test_seqs=data["test_sequences"],
        val_seqs=data["val_sequences"],
        batch_size=batch_size,
    )


def create_model(data, lr: float):
    return SpeciesPredictor(
        n_features=len(data["feature_columns"]),
        n_classes=data["n_classes"],
        lr=lr,
    )


def create_trainer(device: str, max_epochs: int):
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        log_every_n_steps=50,
        deterministic=True,
    )


def train_model(model, data_module, trainer, save_path: Path):
    trainer.fit(model, datamodule=data_module)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(save_path)


def run_lstm(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    val_path=VAL_PATH,
    batch_size=16,
    lr=1e-3,
    max_epochs=2,
):
    data = prepare_data(train_path=train_path, test_path=test_path, val_path=val_path)
    data_module = create_data_module(data, batch_size)
    model = create_model(data, lr)
    trainer = create_trainer(data["device"], max_epochs)
    save_path = Path("trained_model/species_model.ckpt")
    train_model(model, data_module, trainer, save_path)


if __name__ == "__main__":
    run_lstm(batch_size=16, lr=1e-3, max_epochs=2)
