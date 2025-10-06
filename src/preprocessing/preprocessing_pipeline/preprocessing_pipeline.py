from preprocessing.preprocessing_pipeline.data_loader import DataLoader
from preprocessing.preprocessing_steps.outlier_cleaner import SITSOutlierCleaner

class PreprocessingPipeline:
    def __init__(
        self,
        path,
        outlier_cleaner=False,
    ):
        self.path = path
        self.outlier_cleaner = outlier_cleaner

        self.loader = DataLoader()
        self.outlier_cleaner = SITSOutlierCleaner()

        self.df = None

    def run(self, path):
        df = self.loader.load_transform(path)

        if self.anomaly_detection:
            df = self.anomaly_detector.detect(df)

        if self.change_labels:
            df = self.label_transformer.transform(df)

        if self.undersampling:
            df = self.undersampler.apply(df)

        if self.create_indices:
            df = self.index_creator.create(df)

        df = self.feature_engineer.add_features(df)
        return df