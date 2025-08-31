# pipeline.py
from src.datafetcher import TennisDataFetcher
from src.dataprocessor import TennisDataProcessor
from src.datacombiner import TennisDataCombiner
from src.featuresbuilder import FeatureBuilder
from src.model_trainer import ModelTrainer
# from prediction.model_predictor import ModelPredictor

class Pipeline:
    def __init__(self, max_tournaments=1):
        self.data_fetcher = TennisDataFetcher()
        self.data_preprocessor = TennisDataProcessor()
        self.data_combiner = TennisDataCombiner()
        self.feature_builder = FeatureBuilder()
        self.model_trainer = ModelTrainer()
        # self.model_predictor = ModelPredictor()
        self.max_tournaments = max_tournaments

    def run(self):
        self.data_fetcher.get_all_data(max_tournaments=self.max_tournaments)
        self.data_fetcher.close()
        self.data_preprocessor.process_all_data()
        self.data_combiner.combine_data()
        self.feature_builder.build_features()
        self.model_trainer.train_model()
        # prediction = self.model_predictor.make_prediction(trained_model)
        return None # prediction

if __name__ == "__main__":
    pipeline = Pipeline(max_tournaments=10)
    result = pipeline.run()
    print(result)