from pathlib import Path
import pandas as pd
import torch
import pickle
from models.lstm.lstm_utils.data_processor import DataProcessor
from models.lstm.lstm_utils.species_predictor import SpeciesPredictor
from models.lstm.lstm_utils.utility_functions import df_to_sequences
from pipelines.processing.processing_pipeline import ProcessingPipeline
from pipelines.processing.features.basic_features import BasicFeatures
from pipelines.processing.features.temporal_features import TemporalFeatures
from pipelines.processing.features.spectral_indices import CalculateIndices
from pipelines.processing.processing_steps.aggregation import TimeSeriesAggregate
from pipelines.processing.processing_steps.interpolate_nans import InterpolateNaNs
from pipelines.processing.processing_steps.smoothing import Smooth
from pipelines.processing.processing_steps.interpolation import Interpolation

BASE_DIR = Path(__file__).resolve().parents[3]
CKPT_PATH = BASE_DIR / "data/lstm_training/results/epochepoch=89.ckpt"
SCALER_PATH = BASE_DIR / "data/lstm_training/results/encoders/scaler.pkl"
LE_PATH = BASE_DIR / "data/lstm_training/results/encoders/label_encoder.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "data/lstm_training/results/encoders/feature_columns.pkl"

UNSEEN_RAW_PATH = BASE_DIR / "data/val/FINAL_Validierungs_Datensatz.csv"
UNSEEN_PROCESSED_PATH = BASE_DIR / "data/val/val_processed.csv"
OUT_PATH = BASE_DIR / "data/val/val_predictions.csv"


def process_unseen_data(raw_path, processed_path):
    print("🔄 [1/7] Starte Verarbeitung der Rohdaten...")
    steps = [
        BasicFeatures(on=True),
        TimeSeriesAggregate(on=True, freq=2, method="mean"),
        InterpolateNaNs(on=True, method="linear"),
        Smooth(on=True, overwrite=True),
        CalculateIndices(on=True),
        TemporalFeatures(on=True),
        Interpolation(on=True),
    ]
    pipeline = ProcessingPipeline(path=raw_path, steps=steps)
    df_processed = pipeline.run()
    df_processed.to_csv(processed_path, index=False)
    print(f"✅ Rohdaten erfolgreich verarbeitet → gespeichert unter: {processed_path}")
    return processed_path


def predict_unseen(raw_path):
    print("🚀 Starte Vorhersage-Pipeline für unbekannte Daten...\n")

    # Schritt 1: Datenverarbeitung
    processed_path = process_unseen_data(raw_path, UNSEEN_PROCESSED_PATH)

    # Schritt 2: Artefakte laden
    print("\n📦 [2/7] Lade gespeicherte Trainings-Artefakte...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    print(f"✅ LabelEncoder-Klassen: {list(le.classes_)}")
    print(f"✅ Anzahl Feature Columns: {len(feature_columns)}")

    # Schritt 3: Daten laden
    print("\n📂 [3/7] Lade verarbeitete Validierungsdaten...")
    unseen_df = pd.read_csv(processed_path)
    print(f"➡️ Eingelesene Zeilen: {len(unseen_df)} | Spalten: {len(unseen_df.columns)}")

    # Schritt 4: Feature-Auswahl & Encoding
    print("\n🧩 [4/7] Wende gleiche Feature-Logik wie im Training an...")

    exclude_columns = [
        "time",
        "id",
        "disturbance_year",
        "is_disturbed",
        "date_diff",
        "year",
        "doy",
    ]
    categorical_cols = ["season", "is_growing_season", "month_num", "biweek_of_year"]

    ids = unseen_df["id"].copy()
    unseen_df = pd.get_dummies(unseen_df, columns=categorical_cols)
    X_unseen = unseen_df.reindex(columns=feature_columns, fill_value=0)

    print(f"✅ Nach One-Hot-Encoding: {len(X_unseen.columns)} Spalten")

    # Schritt 5: Skalierung
    print("\n📏 [5/7] Skaliere Features mit gespeichertem Scaler...")
    X_unseen[feature_columns] = scaler.transform(X_unseen[feature_columns])
    print("✅ Skalierung abgeschlossen.")

    # 🔹 ID wieder hinzufügen (wichtig für df_to_sequences)
    X_unseen["id"] = ids.values

    # Schritt 6: Sequenzen erstellen
    print("\n📈 [6/7] Erstelle Sequenzen für LSTM...")
    sequences = df_to_sequences(X_unseen, feature_columns, label_column=None)
    print(f"✅ {len(sequences)} Sequenzen erstellt.")

    # Schritt 7: Modell laden und Vorhersagen treffen
    print("\n🧠 [7/7] Lade Modell & führe Vorhersage durch...")
    model = SpeciesPredictor.load_from_checkpoint(
        CKPT_PATH,
        n_features=len(feature_columns),
        n_classes=len(le.classes_),
        lr=1e-3,
    )
    model.eval()
    print("✅ Modell erfolgreich geladen.")

    preds = []
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            output = model(seq_tensor)
            pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1).item()
            preds.append(pred_class)
            if i % 100 == 0:
                print(f"🔹 Fortschritt: {i}/{len(sequences)} Sequenzen verarbeitet...")

    print("\n🔁 Wandle Sequenz-Vorhersagen in ID-basierte Tabelle um...")

    # IDs der Sequenzen extrahieren
    unique_ids = X_unseen["id"].unique()
    if len(unique_ids) != len(preds):
        print(f"⚠️ Warnung: {len(unique_ids)} eindeutige IDs, aber {len(preds)} Vorhersagen!")
        print("   → Überprüfe, ob jede ID genau eine Sequenz ergibt.")

    # Labels zurücktransformieren
    predicted_labels = le.inverse_transform(preds)

    # Ergebnis-DataFrame erstellen
    results_df = pd.DataFrame({
        "id": unique_ids,
        "predicted_species": predicted_labels
    })

    # CSV speichern
    results_df.to_csv(OUT_PATH, index=False)
    print(f"✅ {len(results_df)} Vorhersagen gespeichert unter: {OUT_PATH}")


if __name__ == "__main__":
    predict_unseen(UNSEEN_RAW_PATH)
