# Tree Species Classification & Disturbed Tree Detection using Sentinel-2

This repository contains two Jupyter notebooks focused on the analysis of tree species and detection of potentially disturbed trees using Sentinel-2 satellite imagery. The datasets link the German National Forest Inventory (BWI) 2012 and 2022 with Sentinel-2 observations, providing pixel-level time series data for individual trees.


Tree Species Classification

Objective:
Train machine learning models to classify tree species from Sentinel-2 time series data to support environmental monitoring and sustainable forest management.

## Key Features:

Dataset: Pixel time series of individual trees with linked species labels.

Preprocessing: Cleaning, handling missing values, outlier detection, and interpolation.

Feature Engineering:

Basic and advanced features including vegetation indices.

Date and time features to capture seasonality.

Aggregation to reduce noise and improve feature robustness.

Exploratory Data Analysis (EDA):

Frequency distribution by year, month, species, and tree ID.

Time gap analysis and identification of rare observations.

Visualization of spectral bands and vegetation indices.

Seasonal and cyclic analysis to capture annual vegetation cycles.

Correlation Analysis:

Pearson correlation and top correlations among bands and indices.

Insights into which features are meaningful for modeling.

Disturbance Analysis:

Identification of disturbed trees over time.

Analysis of disturbance frequency per species.

Time Series Analysis:

Aggregation (e.g., 2-week intervals).

Autocorrelation (ACF/PACF) for temporal dependencies.

Anomaly detection using Isolation Forest and Z-score methods.

Dimensionality Reduction:

PCA and t-SNE to visualize separability of species in feature space.

Insights: Soil and beech are separable; conifers overlap.




## Detect Disturbed Trees

Identify potentially disturbed or sick trees using time series of spectral bands and vegetation indices.

Workflow:

Preprocessing:

Interpolation of missing values.

Normalization per species to account for differing value ranges.

Feature Engineering:

Yearly aggregation of bands and indices.

Calculation of key features per tree ID:

Slope of yearly means

Standard deviation of yearly means

Feature Analysis:

Correlation between features and the is_disturbed label.

Visual validation of trees with high feature values to confirm disturbances.

Training Data Preparation:

Disturbed trees: Trees labeled as disturbed (disturbance year ≥ 2017).

Healthy trees: Selected using low values of top features to avoid mislabeling.

Balanced dataset creation for training ML models.

Model Preparation:

Dataset ready for classifier training to detect disturbed trees.

Includes visualization examples of high-feature-value trees.