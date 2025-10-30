`<a id="readme-top"></a>`
`<br />`

<div align="center">
  <img src="../../doc/assets/images/sentinel.jpg" alt="Logo" width="400">
  <h3 align="center">Tree Species Classification</h3>
</div>

<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li><a href="#baseline-model">Baseline Model</a></li>
    <li><a href="#ensemble-models">Ensemble Models</a></li>
    <li><a href="#lstm-model">LSTM Model</a></li>
    <li><a href="#pyts-model">PYTS Model</a></li>
  </ol>
</details>


---
## Baseline Model

The baseline model provides a reference performance level using classical machine learning methods.
It applies a preprocessing and feature engineering pipeline to Sentinel-2 time series data and trains an XGBoost classifier on extracted spectral and temporal features.
**The pipeline includes:**
- Basic feature computation
- Temporal interpolation
- Calculation of spectral indices
- Extraction of temporal statistics and key figures
The trained model is then evaluated using accuracy, F1-score, and confusion matrices.


```python
xgb_baseline_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    random_state=42,
    eval_metric="mlogloss",
    objective="multi:softprob",
    num_class=len(le.classes_),
)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Ensemble Models

### Ensemble Experiments

This section focuses on training **ensemble-based machine learning models** such as **XGBoost** and **Random Forest** with different hyperparameter configurations.
Hyperparameters can be defined via **Grid Search**, allowing systematic experimentation and optimization.

Example setup:

```python
def define_models():
    return [
        (
            RandomForestClassifier,
            {"n_estimators": [2], "max_depth": [15], "min_samples_split": [5]},
        ),
        (
            xgb.XGBClassifier,
            {
                "n_estimators": [10],
                "learning_rate": [0.01],
                "max_depth": [10],
            },
        ),
    ]
```

Each model is trained on preprocessed Sentinel-2 time series data to classify tree species based on spectral and temporal features.

### Ensemble Evaluation

The best-performing model from the grid search is evaluated in detail.
Evaluation includes:

- Calculation of standard classification metrics (accuracy, precision, recall, F1-score).
- Visualization of the confusion matrix to assess class performance.
- Analysis of feature importance to identify the most relevant spectral indices and temporal features.

### Ensemble Pipeline

The full ensemble model pipeline includes:

- Loading and preparing data
- Model training using the selected algorithm
- Evaluation on the validation dataset

Run the pipeline using:

```bash
python run_ensemble_pipeline.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## LSTM Model

### LSTM Experiments

The LSTM (Long Short-Term Memory) model is trained to capture temporal dependencies in Sentinel-2 time series data.
During experimentation, key parameters such as number of epochs, batch size, and sequence length are tuned to optimize performance.

### LSTM Evaluation

The trained LSTM model is evaluated on the validation set using:

- Classification accuracy and loss curves
- Comparison of predicted vs. true species labels
- Optional analysis of feature importance derived from time-step contributions (e.g., via attention or perturbation analysis)

This helps understand which time periods and spectral bands are most relevant for accurate classification.

### LSTM Pipeline

The LSTM training pipeline automates the workflow:

- Load preprocessed time series data
- Train the LSTM model with specified hyperparameters
- Evaluate performance on the validation dataset
  Run the pipeline using:

```bash
python run_lstm_pipeline.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---


## PYTS Model
The PYTS model (Python Time Series classification framework) explores alternative approaches for handling multivariate Sentinel-2 time series data.
**It can incorporate transformations such as:**
- Time-series flattening
- Recurrence plots
- Gramian angular fields
- Shapelet extraction
This model family serves as an experimental extension to the baseline and ensemble models for further performance comparison.

``` python
def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        scaler: object = StandardScaler(),
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.scaler = scaler

```


<p align="right">(<a href="#readme-top">back to top</a>)</p>