from pathlib import Path
import json, getpass
from datetime import datetime

# 1. Projekt-Wurzel korrekt bestimmen
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "data" / "baseline_training" / "baseline_results"

# 2. Benutzerverzeichnis erstellen
user = getpass.getuser()
user_results_dir = RESULTS_DIR / user
user_results_dir.mkdir(parents=True, exist_ok=True)

# 3. Beispieldaten definieren
example_data = {
    "model": "DummyModel",
    "accuracy": 0.95,
    "timestamp": datetime.now().isoformat(),
}

# 4. Datei speichern
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = user_results_dir / f"run_{timestamp}.json"
with open(filename, "w") as f:
    json.dump(example_data, f, indent=4)

print(f"✅ Datei gespeichert: {filename}")
