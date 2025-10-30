# Tree Species Classification & Disturbed Tree Detection using Sentinel-2

This folder contains Jupyter notebooks for analyzing tree species and detecting potentially disturbed (damaged or stressed) trees using **Sentinel-2 satellite imagery**.
The data combines observations from the **German National Forest Inventory (BWI) 2012 and 2022** with Sentinel-2 pixel-level time series for individual trees.

---

## Notebooks Overview

| Folder                     | Notebook                       | Description                                                                                                                                                                                                                                                                                                                                           |
| -------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **data_reduction**   | **data_reduction.ipynb** | Explores strategies for reducing dataset size while maintaining a balance between computational efficiency and model performance. Includes temporal aggregation, yearly and monthly subsampling, and evaluation of optimal time intervals for model training. Results show how combining multiple years improves robustness up to a saturation point. |
| **feature_analysis** | **shap_analysis.ipynb**  | Performs feature importance analysis using SHAP (SHapley Additive exPlanations) based on a trained baseline model. The notebook identifies the most influential spectral bands and vegetation indices for each tree species, both globally and per class, through summary plots, grouped heatmaps, and dependence analyses.                           |

---

## Overview

The notebooks in this folder are part of a broader workflow designed to improve the accuracy and interpretability of models for tree species classification and disturbance detection.
They focus on **optimizing data usage**, **reducing computational costs**, and **understanding model decisions** through explainable AI techniques.

---
