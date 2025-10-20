# Tree Species Classification & Disturbed Tree Detection using Sentinel-2

This folder contains Jupyter notebooks for analyzing tree species and detecting potentially disturbed (damaged or stressed) trees using **Sentinel-2 satellite imagery**.  
The data combines observations from the **German National Forest Inventory (BWI) 2012 and 2022** with Sentinel-2 pixel-level time series for individual trees.

---

## Notebooks Overview

| Notebook | Description |
|-----------|--------------|
| **data_analysis.ipynb** | Focuses on preprocessing, feature engineering, exploratory data analysis (EDA), correlation studies, and dimensionality reduction. |
| **detect_disturbed_trees.ipynb** | Identifies potentially disturbed or unhealthy trees by analyzing temporal changes in spectral bands and vegetation indices. Includes preprocessing, feature extraction, anomaly detection, and preparation for model training. |

---

## Data Analysis

**Objective:**  
Develop and evaluate machine learning models to classify tree species from Sentinel-2 satellite data — supporting sustainable forest management and environmental monitoring.

### Key Steps:
- **Dataset:** Pixel-based time series with species labels linked to BWI 2012 & 2022.  
- **Preprocessing:** Data cleaning, missing value interpolation, and outlier handling.  
- **Feature Engineering:**  
  - Vegetation indices (e.g., NDVI, EVI, NDMI)  
  - Temporal and seasonal features  
  - Aggregation to reduce noise and improve robustness  
- **Exploratory Data Analysis (EDA):**  
  - Distribution by year, month, and species  
  - Time gaps and rare observations  
  - Visualization of spectral bands and vegetation indices  
  - Seasonal and cyclic trends  
- **Correlation Analysis:**  
  - Pearson correlation between features  
  - Identification of the most informative bands and indices  
- **Dimensionality Reduction:**  
  - PCA and t-SNE for visualization of separability among species  
  - Insights: *Soil and beech are well-separated; conifers overlap*  

---

## Disturbed Tree Detection

**Objective:**  
Detect trees showing signs of disturbance or stress using time series of spectral information.

### Workflow:
- **Preprocessing:**  
  - Interpolation of missing values  
  - Normalization per species  
- **Feature Engineering:**  
  - Annual aggregation of spectral indices  
  - Calculation of yearly slope and standard deviation  
- **Feature Analysis:**  
  - Correlation with disturbance labels  
  - Visual inspection of trees with high anomaly indicators  
- **Training Data Preparation:**  
  - Labeling disturbed trees (disturbance ≥ 2017)  
  - Selecting healthy trees with stable spectral patterns  
  - Creating a balanced dataset for classifier training  
- **Model Preparation:**  
  - Dataset ready for ML classification  
  - Includes visualizations of high-feature-value cases  

--- 
