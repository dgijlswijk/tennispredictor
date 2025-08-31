import pandas as pd
import numpy as np
import logging

class TennisDataCombiner:
    """
    A class to create and manage a dataset for training a machine learning model.
    """

    def __init__(self, ):
        """
        Initialize the TennisDataCombiner with data and labels.

        Args:
            data (list or np.ndarray): The input features for training.
            labels (list or np.ndarray): The corresponding labels for the data.
        """

        # Load participants from a CSV file if available
        try:
            self.participants = pd.read_csv("data/processed/participants.csv")
            self.games = pd.read_csv("data/processed/games.csv")
        except FileNotFoundError as e:
            logging.error("Participants or games data file not found. Please ensure the files exist in the specified path.", e)

    def participant_features(self):
        part_df = self.participants.copy()
        np.random.seed(42)
        num_participants = part_df.shape[0]
        random_birthdates = pd.to_datetime(
            np.random.randint(
                pd.Timestamp("1970-01-01").value // 10**9,
                pd.Timestamp("2005-12-31").value // 10**9,
                num_participants
            ),
            unit='s'
        ).normalize()  # Set time to 00:00:00
        part_df['birthdate'] = random_birthdates
        return part_df

    def symmetrize_games(self, df):
        # Symmetrize matches: add a row for each match with home/away swapped and result reversed
        def reverse_result(result):
            # Handles results like '3:1', '2:3', etc.
            if isinstance(result, str) and ':' in result:
                parts = result.split(':')
                return f"{parts[1]}:{parts[0]}"
            return result

        df_train_sym = df.copy()
        swapped = df.copy()
        # Swap all columns containing 'home' with their corresponding 'away' columns and vice versa
        for col in df.columns:
            if 'home' in col:
                away_col = col.replace('home', 'away')
                if away_col in df.columns:
                    swapped[col], swapped[away_col] = df[away_col], df[col]
            elif 'away' in col:
                home_col = col.replace('away', 'home')
                if home_col in df.columns:
                    swapped[col], swapped[home_col] = df[home_col], df[col]

        swapped['result'] = df['result'].apply(reverse_result)

        # Concatenate original and swapped
        df_train_sym = pd.concat([df, swapped], ignore_index=True)
        df_train_sym.reset_index(drop=True, inplace=True)

        logging.info(f"Original matches: {len(df)}, Symmetrized matches: {len(df_train_sym)}")
        return df_train_sym

    def combine_data(self):
        """
        Combine data from participants and games into a single DataFrame.
        """
        game_df = self.games.copy()
        part_df = self.participant_features()[['id', 'name', 'birthdate']]

        # Merge the two DataFrames on 'id'
        combined_df = pd.merge(
            game_df, 
            part_df.add_suffix('_home'),  # Add suffix to all columns in right table
            left_on='home_id', 
            right_on='id_home', 
            suffixes=('', ''), 
            how='inner'
        )
        combined_df = pd.merge(
            combined_df, 
            part_df.add_suffix('_away'),  # Add suffix to all columns in right table
            left_on='away_id', 
            right_on='id_away', 
            suffixes=('', ''), 
            how='inner'
        )

        sym_df = self.symmetrize_games(combined_df)

        sym_df.to_csv("data/processed/combined.csv", index=False)
        
        logging.info("Features saved to data/processed/combined.csv")

        return sym_df

if __name__ == "__main__":
    tts = TennisDataCombiner()
    tts.combine_data()