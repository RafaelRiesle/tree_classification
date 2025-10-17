from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from models.lstm.lstm_utils.species_data_module import SpeciesDataModule
from models.lstm.lstm_utils.predictor_species import SpeciesPredictor
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


def create_trainer(device: str, max_epochs: int, checkpoint_callback):
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        log_every_n_steps=50,
        deterministic=True,
        callbacks=[checkpoint_callback],
    )


def train_model(model, data_module, trainer):
    trainer.fit(model, datamodule=data_module)

    # Finde den ModelCheckpoint Callback
    checkpoint_callback = None
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint):
            checkpoint_callback = cb
            break

    if checkpoint_callback is None:
        raise RuntimeError("Kein ModelCheckpoint Callback gefunden!")

    # best_model_path zurückgeben
    best_model_path = checkpoint_callback.best_model_path
    return best_model_path



def run_lstm(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    val_path=VAL_PATH,
    batch_size=16,
    lr=1e-3,
    max_epochs=2,
):
    # Daten vorbereiten
    data = prepare_data(train_path=train_path, test_path=test_path, val_path=val_path)
    data_module = create_data_module(data, batch_size)
    model = create_model(data, lr)

    # ModelCheckpoint für bestes Modell nach Val-Loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="trained_model/",
        filename="species_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    trainer = create_trainer(data["device"], max_epochs, checkpoint_callback)

    # Training starten und bestes Modell erhalten
    best_model_path = train_model(model, data_module, trainer)
    print(f"Bestes Modell gespeichert unter: {best_model_path}")



if __name__ == "__main__":
    run_lstm(batch_size=32, lr=1e-3, max_epochs=200)
