<a id="readme-top"></a>
<br />
<div align="center">
  <img src="../doc/assets/images/sentinel.jpg" alt="Logo" width="400">
  <h3 align="center">Tree Species Classification</h3>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#models">Models</a>
      <ul>
        <li><a href="#ensemble-models">Ensemble Models</a></li>
        <li><a href="#lstm-model">LSTM Model</a></li>
      </ul>
    </li>
    <li>
      <a href="#pipelines">Pipelines</a>
      <ul>
        <li><a href="#preprocessing">Preprocessing</a></li>
        <li><a href="#processing">Processing</a></li>
        <li><a href="#training">Training</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>


---

## Models

This section contains the implementation and training scripts for various models used in the **Tree Species Classification** task.  
Each model is trained using Sentinel-2 time series data and derived vegetation indices to identify dominant tree species in German forests.

A more detailed description of the model architectures, parameters, and evaluation metrics can be found in the dedicated [Models README](models/README.md).

---

### Ensemble Models

The ensemble models combine multiple machine learning algorithms (e.g., Random Forest, Gradient Boosting, XGBoost) to improve prediction accuracy and robustness.  
These models leverage the diversity of different learners to minimize classification errors and handle the high variability of spectral data.

---

### LSTM Model

The LSTM (Long Short-Term Memory) model is designed to capture **temporal dependencies** within the Sentinel-2 time series data.  
It analyzes the sequential patterns of spectral features and vegetation indices to model seasonal and phenological trends for improved species classification.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Pipelines

The repository includes modular pipelines to handle data preprocessing, feature processing, and model training.  
Each pipeline can be executed independently to prepare and process the data efficiently.

---

### Preprocessing

```bash
python3 run_preprocessing_pipeline.py
```

The preprocessing pipeline performs initial preparation of the raw dataset.
Main tasks include:

Splitting the dataset into training, validation, and test subsets.

Handling missing values and cleaning inconsistencies.

Detecting and optionally removing outliers to improve model stability.

Ensuring that the temporal order of observations is preserved.


### Processing
```bash
python3 run_processing_pipeline.py```
```

The processing pipeline focuses on feature extraction and index computation.
Key operations include:

Calculating vegetation indices (e.g., NDVI, EVI, NDMI, NBR).

Aggregating spectral data over time windows (e.g., monthly or seasonal means).

Creating derived features that capture temporal and spectral variability.

Preparing the processed feature set for downstream model training.

### Training
```bash
python3 run_training_pipeline.py```
```
The training pipeline executes the full model training and evaluation workflow, including:

Loading preprocessed and processed data.

Running cross-validation and hyperparameter tuning.

Evaluating model performance across multiple metrics (accuracy, F1-score, etc.).

Exporting trained models and performance reports for further analysis.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Roadmap

- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

<p align="right">(<a href="#readme-top">back to top</a>)</p>