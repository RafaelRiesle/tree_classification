import numpy as np
import pandas as pd
from general_utils.constants import spectral_bands
from utils.data_loader import DataLoader
from utils.sits_outlier_cleaner import SITSOutlierCleaner


def run_detect_outlier():
    INPUT_PATH = "data/baseline_training/trainset.csv"
    OUTPUT_PATH = "data/processed/cleaned_trainset.csv"
    dataloader = DataLoader()

    # === Load dataset ===
    print("Loading dataset...")
    df = dataloader.load_transform(INPUT_PATH)

    df_sample = dataloader.get_sample(df, n_ids=50)

    # === Outlier cleaning ===
    print("Removing outliers from the dataset...")
    cleaner = SITSOutlierCleaner(contamination=0.05)
    cleaned_df = cleaner.fit_transform(df_sample, spectral_bands)

    # === Save cleaned dataset ===
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")
