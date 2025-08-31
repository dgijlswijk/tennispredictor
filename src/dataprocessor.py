import logging
from typing import Optional, Dict, Any
import pandas as pd
import json
import os   
import re

class TennisDataProcessor:
    def __init__(self):
        pass

    def process_cuptree_json(self, cuptree: Dict[str, Any]) -> pd.DataFrame:
        """Extracts relevant columns from a cuptree JSON into a DataFrame with error handling."""
        
        try:
            columns = [
                'id',
                'name',
                'tournament',
                'rounds',
            ]

            df = pd.DataFrame(cuptree)
            # Only keep the specified columns if they exist
            df = df[[col for col in columns if col in df.columns]]

            return df
        except Exception as e:
            logging.error(f"Error processing cuptree JSON: {e}")
            return pd.DataFrame()
    
    def get_all_participants(self, cuptrees: pd.DataFrame) -> Any:
        """
        Extracts all participants from the 'rounds' column of the cuptree DataFrame.
        Returns a DataFrame with all participants flattened.
        """
        participants_list = []

        for rounds in cuptrees['rounds']:
            for round_item in rounds:
                blocks = round_item.get('blocks', [])
                for block in blocks:
                    participants = block.get('participants', {})
                    for participant in participants:
                        team = participant.get('team', {})
                        team['winner'] = participant.get('winner', None)
                        team['order'] = participant.get('order', None)
                        team['teamSeed'] = participant.get('teamSeed', None)
                        participants_list.append(team)
        
        return pd.DataFrame(participants_list)[['name', 'slug', 'shortName', 'gender', 'nameCode', 'ranking', 'disabled', 'national', 'id']].drop_duplicates()
                  
    def map_round_description(self, round_description: str) -> str:
        """
        Maps a round description dictionary to a human-readable string.
        Raises a ValueError if the round_description is not recognized.
        """
        if not round_description:
            return ""
        
        if re.search(r'qualification|qualifying', round_description, re.IGNORECASE):
            return 'Q'
        
        # Map round codes to simplified descriptions
        round_map = {
            'R128': 'R128',
            'R64': 'R64',
            'R32': 'R32',
            'R16': 'R16',
            'Quarterfinals': 'QF',
            'Quarterfinal': 'QF',
            'Semifinals': 'SF',
            'Semifinal': 'SF',
            'Final': 'F',
            '1/32': 'R32',
            '1/16': 'R16',
            '1/8': 'R8',
            'Qualification': 'Q',
            'Qualification round 1': 'Q',
            'Qualification round 2': 'Q',
            'Qualification Final': 'Q',
            'Qualification final': 'Q',
            'Round of 128': 'R128',
            'Round of 64': 'R64',
            'Round of 32': 'R32',
            'Round of 16': 'R16',
        }
        if round_description not in round_map:
            raise ValueError(f"Unknown round description: {round_description}")
        return round_map[round_description]
    
    def validate_score_format(self, score: str) -> bool:
        # Define a regular expression pattern to match scores in the format "0:3", "1:2", etc.
        pattern = re.compile(r'^[0-3]:[0-3]$')

        if score and pattern.match(score):
            return True
        else:
            return False

    def extract_games_from_cuptree(self, cuptrees: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and flattens all games from the 'rounds' column of the cuptree DataFrame.
        Returns a DataFrame with one row per game, including home and away participant IDs.
        """
        games_list = []

        for _, cuptree in cuptrees.iterrows():
            tournamentName = cuptree['tournament']['name']
            uniqueTournament = cuptree['tournament']['uniqueTournament']['name']
            rounds = cuptree['rounds']
            if not isinstance(rounds, list):
                continue
            for round_item in rounds:
                round_description = round_item.get('description', {})
                blocks = round_item.get('blocks', [])
                for block in blocks:
                    if block.get('result') and block.get('result').lower() in ['retired', 'walkover', '0:0']:
                        continue
                    elif block.get('result') in ['home won', 'away won', 'on-going']:
                        block['result'] = block['homeTeamScore'] + ':' + block['awayTeamScore']
                    if not self.validate_score_format(block.get('result')):
                        logging.error(f"Invalid score format: {block.get('result')}")
                    participants = block.get('participants', [])
                    block_copy = block.copy()
                    if len(participants) < 1:
                        continue
                    block_copy['home_id'] = participants[0]['team'].get('id')
                    if len(participants) == 1:
                        block_copy['away_id'] = None
                    else:
                        block_copy['away_id'] = participants[1]['team'].get('id')
                    block_copy['round_description'] = self.map_round_description(round_description)
                    block_copy['tournamentName'] = tournamentName
                    block_copy['uniqueTournament'] = uniqueTournament
                    
                    games_list.append(block_copy)

        if not games_list:
            return pd.DataFrame()

        games_df = pd.json_normalize(games_list)

        games_df['seriesStartDate'] = pd.to_datetime(games_df.get('seriesStartDateTimestamp', None), unit='s', errors='coerce')

        return games_df[['finished', 'result', 'homeTeamScore',
       'awayTeamScore', 
       'id', 'events', 'seriesStartDate',
       'home_id', 'away_id', 'round_description', 'tournamentName', 'uniqueTournament']]
    
    def process_all_data(self, max_cuptrees: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Processes all cuptree JSON files and returns a dictionary with DataFrames for:
        - tournaments
        - participants
        - games

        Args:
            max_cuptrees: Maximum number of cuptree files to process. If None, process all.
        """
        data_dir = os.path.join('data', 'raw')
        cuptree_files = [f for f in os.listdir(data_dir) if f.startswith('cuptrees') and f.endswith('.json')]
        if not cuptree_files:
            raise FileNotFoundError("No files starting with 'cuptrees' found in data/raw")

        if max_cuptrees is not None:
            cuptree_files = cuptree_files[:max_cuptrees]

        participants_list = []
        games_list = []

        for cuptree_file in cuptree_files:
            logging.info(f"Processing cuptree file: {cuptree_file}")
            with open(os.path.join(data_dir, cuptree_file), 'r', encoding='utf-8') as f:
                cuptree = json.load(f)
            processed_cuptree = self.process_cuptree_json(cuptree)
            participants_df = self.get_all_participants(processed_cuptree)
            games_df = self.extract_games_from_cuptree(processed_cuptree)
            participants_list.append(participants_df)
            games_list.append(games_df)

        participants_df = pd.concat(participants_list, ignore_index=True) if participants_list else pd.DataFrame()
        games_df = pd.concat(games_list, ignore_index=True) if games_list else pd.DataFrame()

        participants_df = participants_df.drop_duplicates().reset_index(drop=True)

        # Save participants and games DataFrames separately as CSV files
        participants_df.drop_duplicates().to_csv(os.path.join('data', 'processed', 'participants.csv'), index=False)
        games_df.to_csv(os.path.join('data', 'processed', 'games.csv'), index=False)

        return {
            "participants": participants_df,
            "games": games_df
        }
    

if __name__ == "__main__":
    processor = TennisDataProcessor()
    df = processor.process_all_data()