# Initial SITS Pipeline

Dieses Projekt enthält eine Pipeline zur Verarbeitung und Bereinigung von Satellitendaten (SITS). Die Pipeline lädt Rohdaten, erstellt Trainings-, Validierungs- und Test-Splits und entfernt Ausreißer in den Daten.

---

Die Pipeline kann direkt über Python gestartet werden:

<pre class="overflow-visible!" data-start="1308" data-end="1359"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python initial_pipeline/run_pipeline.py</span></span></code></div></div></pre>

## Funktionsweise

1. **Daten laden & transformieren**
   Die Rohdaten aus  `data/raw/raw_trainset.csv` werden geladen und vorverarbeitet.
2. **Train/Test/Validation Split**
   Das Dataset wird in Trainings-, Test- und Validierungssplits aufgeteilt (Standard: 70/20/10).
3. **Outlier Detection**
   Jeder Split wird mit dem **`SITSOutlierCleaner` bereinigt, um Ausreißer zu entfernen.
   Die bereinigten Splits werden in** `data/processed/` gespeichert.
