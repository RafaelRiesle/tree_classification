import pandas as pd


class BasicDataAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_dtypes(self):
        return self.df.dtypes

    def get_description(self):
        return self.df.describe()

    def get_num_rows(self):
        return len(self.df)

    def get_num_cols(self):
        return len(self.df.columns)

    def get_missing_counts(self):
        return (
            self.df.isna()
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "column_name", 0: "missing_count"})
        )

    def get_unique_count(self, column: str):
        return self.df[column].nunique()

    def get_min(self, column: str):
        return self.df[column].min()

    def get_max(self, column: str):
        return self.df[column].max()

    def check_id_species_uniqueness(self) -> bool:
        return self.df.groupby("id")["species"].nunique().le(1).all()

    def show_overview(self):
        print("Number of rows:", self.get_num_rows())
        print("Number of columns:", self.get_num_cols())
        print("Unique IDs:", self.get_unique_count("id"))
        print("Unique species:", self.get_unique_count("species"))
        print("Earliest time:", self.get_min("time"))
        print("Latest time:", self.get_max("time"))
        print("Is id + species unique?", self.check_id_species_uniqueness())
