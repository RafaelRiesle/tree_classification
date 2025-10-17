# Tree Species Classification with LSTM

## Project Overview

This project implements a time-series LSTM model to classify tree species based on processed sensor data. The pipeline is modular, separating data processing, model training, and evaluation, making it easy to extend and reuse.

The pipeline includes:

Data preprocessing (handling missing values, feature scaling, label encoding, sequence creation)

LSTM model training with PyTorch Lightning

Model evaluation using accuracy metrics, confusion matrix, and permutation feature importance

Saving evaluation results and plots for reproducibility



## Project Structure
src/
├── models/
│   └── lstm/
│       ├── pipeline/
│       │   ├── data_processor.py        # Class for data loading, cleaning, and sequence creation
│       │   ├── data_module.py           # PyTorch Lightning DataModule
│       │   ├── model.py                 # LSTM model class
│       │   └── run_lstm_training.py     # Script to run full training and evaluation
│       ├── experiments/
│       │   └── lstm_trainer.py          # Class to train LSTM model
│       └── evaluation/
│           └── model_evaluator.py      # Class for evaluating model & saving plots
├── data/
│   └── processed/                       # Processed CSV datasets
└── README.md
