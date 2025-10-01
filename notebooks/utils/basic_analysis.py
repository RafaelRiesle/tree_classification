import pandas as pd


class BasicDataAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_dtypes(self):
        return self.df.dtypes
    
    def get_desricption(self):
        return self.df.describe()

    def get_num_rows(self):
        return len(self.df)

    def get_num_cols(self):
        return len(self.df.columns)

    def get_missing_counts(self):
        return self.df.isna().sum().sort_values(ascending=False)
