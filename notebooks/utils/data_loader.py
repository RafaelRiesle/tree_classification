import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self):
        pass  
    
    def load_transform(self, path):
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d") 
        df = df.sort_values(["id", "time"])
        return df
    
    def date_feature_extraction(self, df):
        df = df.copy()
        df["month_num"] = df["time"].dt.month
        df["year"] = df["time"].dt.year
        seasons = np.array(["Winter", "Spring", "Summer", "Autumn"])
        df["season"] = seasons[((df["month_num"] % 12) // 3)]
        df["date_diff"] = df.groupby("id")["time"].diff().dt.days
        return df

    def feature_extraction(self,df):
        df["disturbed"] = df["disturbance_year"].apply(lambda x: False if x == 0 else True)
        return df
