import pandas as pd
import numpy as np


class CalculateIndices:
    @staticmethod
    def calculate_ndvi(df: pd.DataFrame):
        return (df["b8"] - df["b4"]) / (df["b8"] + df["b4"])

    @staticmethod
    def calculate_gndvi(df: pd.DataFrame):
        return (df["b7"] - df["b3"]) / (df["b7"] + df["b3"])

    @staticmethod
    def calculate_wdvi(df: pd.DataFrame):
        return df["b8"] - 0.5 * df["b4"]

    @staticmethod
    def calculate_tndvi(df: pd.DataFrame):
        val = (df["b8"] - df["b4"]) / (df["b8"] + df["b4"]) + 0.5
        val = np.clip(val, a_min=0, a_max=None)
        return np.sqrt(val)

    @staticmethod
    def calculate_savi(df: pd.DataFrame, L=0.5):
        return ((df["b8"] - df["b4"]) / (df["b8"] + df["b4"] + L)) * (1 + L)

    @staticmethod
    def calculate_ipvi(df: pd.DataFrame):
        return df["b8"] / (df["b8"] + df["b4"])

    @staticmethod
    def calculate_mcari(df: pd.DataFrame):
        return ((df["b5"] - df["b4"]) - 0.2 * (df["b5"] - df["b3"])) * (
            df["b5"] / df["b4"]
        )

    @staticmethod
    def calculate_reip(df: pd.DataFrame):
        reip = (700 + 40 * (((df["b4"] + df["b7"]) / 2) - df["b5"]) / (
            df["b6"] - df["b5"])
        )
        reip = reip.replace([np.inf, -np.inf], np.nan)
        return reip

    @staticmethod
    def calculate_masvi2(df: pd.DataFrame):
        return (2 * df["b8"] - 1 - np.sqrt((2 * df["b8"] + 1) ** 2 - 8)) / 2

    @staticmethod
    def calculate_dvi(df: pd.DataFrame):
        return df["b8"] - df["b4"]

    @staticmethod
    def add_all_indices(df: pd.DataFrame):
        df = df.copy()
        df["ndvi"] = CalculateIndices.calculate_ndvi(df)
        df["gndvi"] = CalculateIndices.calculate_gndvi(df)
        df["wdvi"] = CalculateIndices.calculate_wdvi(df)
        df["tndvi"] = CalculateIndices.calculate_tndvi(df)
        df["savi"] = CalculateIndices.calculate_savi(df)
        df["ipvi"] = CalculateIndices.calculate_ipvi(df)
        df["mcari"] = CalculateIndices.calculate_mcari(df)
        df["reip"] = CalculateIndices.calculate_reip(df)
        df["masvi2"] = CalculateIndices.calculate_masvi2(df)
        df["dvi"] = CalculateIndices.calculate_dvi(df)
        return df
