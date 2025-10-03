import pandas as pd


class BasicDataAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_dtypes(self):
        """Return data types of all columns."""
        return self.df.dtypes

    def get_description(self):
        """Return descriptive statistics for numeric columns."""
        return self.df.describe()

    def get_num_rows(self):
        """Return the number of rows in the DataFrame."""
        return len(self.df)

    def get_num_cols(self):
        """Return the number of columns in the DataFrame."""
        return len(self.df.columns)

    def get_missing_counts(self):
        """Return a DataFrame with missing value counts per column, sorted descending."""
        return (
            self.df.isna()
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "column_name", 0: "missing_count"})
        )

    def get_unique_count(self, column: str):
        """Return the number of unique values in a column."""
        return self.df[column].nunique()

    def get_min(self, column: str):
        """Return the minimum value of a column."""
        return self.df[column].min()

    def get_max(self, column: str):
        """Return the maximum value of a column."""
        return self.df[column].max()

    def check_id_species_uniqueness(self) -> bool:
        return self.df.groupby("id")["species"].nunique().le(1).all()
