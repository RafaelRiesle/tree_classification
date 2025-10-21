import pandas as pd
import numpy as np


class CalculateIndices:
    def __init__(self, on=True):
        self.on = on

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
        mcari = ((df["b5"] - df["b4"]) - 0.2 * (df["b5"] - df["b3"])) * (
            df["b5"] / df["b4"]
        )
        mcari = mcari.replace([np.inf, -np.inf], np.nan)
        return mcari

    @staticmethod
    def calculate_reip(df: pd.DataFrame):
        reip = 700 + 40 * (((df["b4"] + df["b7"]) / 2) - df["b5"]) / (
            df["b6"] - df["b5"]
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
    def calculate_ndmi(df: pd.DataFrame):
        return (df["b8"] - df["b11"]) / (df["b8"] + df["b11"])

    @staticmethod
    def calculate_nbr(df: pd.DataFrame):
        return (df["b8"] - df["b12"]) / (df["b8"] + df["b12"])

    @staticmethod
    def calculate_ndwi(df: pd.DataFrame):
        return (df["b3"] - df["b8"]) / (df["b3"] + df["b8"])

    @staticmethod
    def calculate_mtci(df: pd.DataFrame):
        mtci = (df["b6"] - df["b5"]) - 0.5 * (df["b4"] - df["b5"])
        return mtci.replace([np.inf, -np.inf], np.nan)
    

    @staticmethod
    def calculate_rendvi(df: pd.DataFrame):
        return (df["b7"] - df["b5"]) / (df["b7"] + df["b5"])

    def run(self, df: pd.DataFrame):
        if not self.on:
            return df

        df = df.copy()

        df["ndvi"] = self.calculate_ndvi(df)
        df["gndvi"] = self.calculate_gndvi(df)
        df["wdvi"] = self.calculate_wdvi(df)
        df["tndvi"] = self.calculate_tndvi(df)
        df["savi"] = self.calculate_savi(df)
        df["ipvi"] = self.calculate_ipvi(df)
        df["mcari"] = self.calculate_mcari(df)
        df["reip"] = self.calculate_reip(df)
        df["masvi2"] = self.calculate_masvi2(df)
        df["dvi"] = self.calculate_dvi(df)

        # Sentinel-2 Indizes
        df["ndmi"] = self.calculate_ndmi(df)
        df["nbr"] = self.calculate_nbr(df)
        df["ndwi"] = self.calculate_ndwi(df)
        df["mtci"] = self.calculate_mtci(df)
        df["rendvi"] = self.calculate_rendvi(df)

        return df
