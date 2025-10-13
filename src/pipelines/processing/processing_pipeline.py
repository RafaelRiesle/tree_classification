import pandas as pd


class ProcessingPipeline:
    def __init__(self, steps=None, path: str = None):
        """
        Example: steps = [BasicFeatures(), TemporalFeatures()]
        """
        self.steps = steps or []
        self.path = path
        self.df = None

        if self.path is not None:
            self.df = pd.read_csv(self.path, parse_dates=["time"])

    def add_step(self, step):
        self.steps.append(step)

    def run(self, df: pd.DataFrame = None) -> pd.DataFrame:
        df = (df or self.df).copy()

        for step in self.steps:
            df = step.run(df)
        return df