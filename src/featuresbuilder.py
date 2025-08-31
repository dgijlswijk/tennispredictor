# features/feature_builder.py
import pandas as pd
import logging
from datetime import datetime
import numpy as np 

# TODO should be in config but for now stays here
ordinal_mapping = {
    'Q1': 1,
    'Q2': 2,
    'Q': 3,
    'R128': 4,
    'R64': 5,
    'R32': 6,
    'R16': 7,
    'R8': 8,
    'QF': 9,
    'SF': 10,
    'F': 11
}

tournament_surfaces = {
    'Australian Open': 'Hard',
    'Roland Garros': 'Clay'
}

class FeatureBuilder:
    def __init__(self):
        self.combined_data = pd.read_csv('data/processed/combined.csv')

    def define_label(self, result: str) -> str:
        # Define the label
        if isinstance(result, str) and ':' in result:
            parts = result.split(':')
            if int(parts[0]) > int(parts[1]):
                return 'home'
            else:
                return 'away'
        else:
            logging.error(f"Invalid result format: {result}")
            raise ValueError(f"Invalid result format: {result}")
        
    def player_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        features['id_home'] = data['id_home']
        features['id_away'] = data['id_away']
        features['age_home'] = (pd.to_datetime(data['seriesStartDate']) - pd.to_datetime(data['birthdate_home'])).dt.days.div(365.25)
        features['age_away'] = (pd.to_datetime(data['seriesStartDate']) - pd.to_datetime(data['birthdate_away'])).dt.days.div(365.25)
        return features


    def build_features(self) -> pd.DataFrame:
        # Extract relevant features from the combined data
        data = self.combined_data.copy()
        data.sort_values(by="seriesStartDate", inplace=True)
        features = pd.DataFrame()

        features['result'] = data['result'].apply(self.define_label) # train label

        features = pd.concat([features, self.player_features(data)], axis=1)

        try:
            features['surface'] =  data['uniqueTournament'].map(tournament_surfaces)
        except KeyError as e:
            logging.error(f"Unknown tournament surface: {e}")
            raise ValueError(f"Unknown tournament surface: {e}")
        try:
            features['round_ordinal'] = data['round_description'].map(ordinal_mapping)
        except KeyError as e:
            logging.error(f"Unknown round description: {e}")
            raise ValueError(f"Unknown round description: {e}")
        
        features['month'] = pd.to_datetime(data['seriesStartDate']).dt.month
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        features = pd.get_dummies(features, columns=['month', 'surface'])

        features.to_csv('data/processed/features.csv', index=False)
        return features
    
if __name__ == "__main__":
    feature_builder = FeatureBuilder()
    features = feature_builder.build_features()
    print("Features built successfully.")