
# Baseline Models Project

This repository contains utilities and pipelines to train, evaluate, and manage baseline machine learning models. It provides tools to systematically track experiments, compute metrics, visualize results, and aggregate baseline model performance.

Folder Structure

Baseline_models/
│
├── baseline_utils/
│   ├── baseline_model_manager.py
│   ├── baseline_pipeline.py
│   └── train_test_split.py
│
├── evaluation/
│   └── evaluation_utils.py
│
├── pipelines/
│   └── pipeline_generic.py
│
├── data/
│   ├── raw/
│   │   └── raw_trainset.csv
│   └── baseline_training/
│       └── baseline_results/
│
├── scripts/
│   └── run_baselines.py          
│
└── README.md


## Key Components

### `baseline_utils/baseline_model_manager.py`

* Central manager for baseline models.
* Features:
  * Train and predict models.
  * Compute metrics: accuracy, precision, recall, F1, confusion matrix, classification report.
  * Extract feature importances.
  * Save/load models in JSON format.
  * Aggregate all experiment results.

### `baseline_utils/baseline_pipeline.py`

* Base preprocessing pipeline (`BasePipeline`) for data transformations.

### `baseline_utils/train_test_split.py`

* Load datasets and split into train, test, and validation sets.

### `pipelines/pipeline_generic.py`

* Generic pipeline class to run multiple baseline models.
* Supports defining models with hyperparameters and automatically saving results.

### `evaluation/evaluation_utils.py`

* Functions to:
  * Select the best model based on a metric.
  * Print classification reports.
  * Plot confusion matrices and feature importances.

---

## How to Run

1. Prepare your dataset in `data/raw/raw_trainset.csv`.
2. Split dataset using `DatasetSplitLoader`.
3. Define baseline models:

<pre class="overflow-visible!" data-start="2386" data-end="2533"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>from</span><span> sklearn.ensemble </span><span>import</span><span> RandomForestClassifier

models = [
    (RandomForestClassifier, {</span><span>"n_estimators"</span><span>: </span><span>10</span><span>, </span><span>"max_depth"</span><span>: </span><span>3</span><span>}),
]
</span></span></code></div></div></pre>

4. Run the pipeline:

<pre class="overflow-visible!" data-start="2557" data-end="2728"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>from</span><span> pipelines.pipeline_generic </span><span>import</span><span> GenericPipeline

pipeline = GenericPipeline(target_col=</span><span>"species"</span><span>)
df_results = pipeline.run(train_df, test_df, models)
</span></span></code></div></div></pre>

5. Evaluate and select the best model:

<pre class="overflow-visible!" data-start="2770" data-end="2984"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>from</span><span> evaluation.evaluation_utils </span><span>import</span><span> get_best_model, evaluate_and_report

best_run_id, best_model, df_sorted = get_best_model(baseline, metric=</span><span>"accuracy"</span><span>)
evaluate_and_report(baseline, best_run_id)
</span></span></code></div></div></pre>

---

## JSON Result Structure

Each run is saved as a JSON file:

<pre class="overflow-visible!" data-start="3052" data-end="3576"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"run_id"</span><span>:</span><span></span><span>"20251006_123456"</span><span>,</span><span>
  </span><span>"timestamp"</span><span>:</span><span></span><span>"2025-10-06T12:34:56"</span><span>,</span><span>
  </span><span>"model"</span><span>:</span><span></span><span>"RandomForestClassifier"</span><span>,</span><span>
  </span><span>"hyperparams"</span><span>:</span><span></span><span>{</span><span>"n_estimators"</span><span>:</span><span></span><span>10</span><span>,</span><span></span><span>"max_depth"</span><span>:</span><span></span><span>3</span><span>}</span><span>,</span><span>
  </span><span>"metrics"</span><span>:</span><span></span><span>{</span><span>
    </span><span>"accuracy"</span><span>:</span><span></span><span>0.95</span><span>,</span><span>
    </span><span>"precision_macro"</span><span>:</span><span></span><span>0.96</span><span>,</span><span>
    </span><span>"recall_macro"</span><span>:</span><span></span><span>0.94</span><span>,</span><span>
    </span><span>"f1_macro"</span><span>:</span><span></span><span>0.95</span><span>,</span><span>
    </span><span>"classification_report"</span><span>:</span><span></span><span>{</span><span> ... </span><span>}</span><span>,</span><span>
    </span><span>"confusion_matrix"</span><span>:</span><span></span><span>[</span><span> ... </span><span>]</span><span>,</span><span>
    </span><span>"feature_importances"</span><span>:</span><span></span><span>[</span><span>{</span><span>"feature"</span><span>:</span><span></span><span>"petal_length"</span><span>,</span><span></span><span>"importance"</span><span>:</span><span></span><span>0.4</span><span>}</span><span>,</span><span> ...</span><span>]</span><span>
  </span><span>}</span><span>,</span><span>
  </span><span>"features"</span><span>:</span><span></span><span>[</span><span>"sepal_length"</span><span>,</span><span></span><span>"sepal_width"</span><span>,</span><span></span><span>"petal_length"</span><span>,</span><span></span><span>"petal_width"</span><span>]</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---
