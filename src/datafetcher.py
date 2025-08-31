import logging
from typing import Optional, Dict, Any, List
import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO)

class TennisDataFetcher:
    """
    Collects tennis data from the SofaScore API.
    """
    def __init__(self):
        """
        Initializes the TennisDataFetcher object.
        """
        self.base_url = "https://www.sofascore.com/api/v1"

        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            logging.error(f"Failed to initialize Selenium WebDriver: {e}")

    
    def _call_using_selenium(self, endpoint: str) -> Dict[str, Any]:
        """
        Uses Selenium to fetch the page source from a given URL.
        This is useful for pages that require JavaScript to render content.
        Optimized for resource management and robustness.
        """
        url = self.base_url + endpoint
        
        self.driver.get(url)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        body = soup.body
        if body:
            # Extract text from <pre> if present, else fallback to body text
            pre = body.find('pre')
            if pre:
                body_content = pre.get_text()
            else:
                body_content = body.get_text()
        else:
            body_content = ''
        try:
            return json.loads(body_content)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
            return {}


    def _validate_response(self, data: Any, required_keys: List[str], context: str = "") -> None:
        """
        Validates that the API response is a dictionary and contains the required keys.
        Raises ValueError or KeyError if validation fails.
        """
        if not isinstance(data, dict):
            raise ValueError(f"API response for {context or 'request'} is not a dictionary: {type(data)}")
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing expected key '{key}' in API response for {context or 'request'}")

    def get_tournaments(self, save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns a list of ATP tournaments with selected fields.
        Optionally saves the data if save_dir is provided.
        """
        data = self._call_using_selenium(
            endpoint="/config/default-unique-tournaments/NL/tennis"
        )
        self._validate_response(data, ["uniqueTournaments"], context="get_tournaments")
        tournaments = data.get("uniqueTournaments", [])
        relevant_fields = ['name', 'slug', 'category', 'tennisPoints', 'id']
        filtered = [
            {k: t.get(k) for k in relevant_fields}
            for t in tournaments
            if t.get("category", {}).get("name") == "ATP"
        ]
        if save_dir:
            self.save_data(filtered, os.path.join(save_dir, "tournaments.json"))
        return filtered

    def get_seasons(self, tournament_id: int, save_dir: str) -> List[Dict[str, Any]]:
        """
        Returns a list of seasons for a given tournament ID.
        Optionally saves the data if save_dir is provided.
        """
        season_file = os.path.join(save_dir, f"seasons_{tournament_id}.json")
        if os.path.exists(season_file):
            with open(season_file, "r", encoding="utf-8") as f:
                seasons = json.load(f)
            return seasons
        
        endpoint = f"/unique-tournament/{tournament_id}/seasons"
        data = self._call_using_selenium(endpoint=endpoint)
        self._validate_response(data, ["seasons"], context="get_seasons")
        seasons = data.get("seasons", [])
        if save_dir:
            self.save_data(seasons, os.path.join(save_dir, f"seasons_{tournament_id}.json"))
        return seasons

    def get_cuptrees(self, tournament_id: int, season_id: int, save_dir: str) -> Dict[str, Any]:
        """
        Returns the cup trees for a specific tournament season.
        Optionally saves the data if save_dir is provided.
        """
        cuptrees_file = os.path.join(save_dir, f"cuptrees_{tournament_id}_{season_id}.json")
        if os.path.exists(cuptrees_file):
            with open(cuptrees_file, "r", encoding="utf-8") as f:
                cuptrees = json.load(f)
            return cuptrees
        
        endpoint = f"/unique-tournament/{tournament_id}/season/{season_id}/cuptrees"
        data = self._call_using_selenium(endpoint=endpoint)
        self._validate_response(data, ["cupTrees"], context="get_cuptrees")
        cuptrees = data.get("cupTrees", {})
        if save_dir:
            self.save_data(cuptrees, os.path.join(save_dir, f"cuptrees_{tournament_id}_{season_id}.json"))
        return cuptrees

    def get_players(self) -> List[Dict[str, Any]]:
        """
        Returns players data.
        """
        # Placeholder for future implementation
        return []

    def save_data(self, data: Any, filename: str) -> None:
        """
        Saves the given data to a file in JSON format.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info("Data successfully saved to %s", filename)
        except Exception as e:
            logging.error("Failed to save data to %s: %s", filename, e)

    def get_all_data(self, max_tournaments: Optional[int] = None, save_dir: str = os.path.join("data", "raw")) -> Dict[str, Any]:
        """
        Collects and saves data for up to max_tournaments ATP tournaments.
        Data is saved in separate files for tournaments, seasons, and cuptrees.
        Returns a dictionary with all collected data.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        all_data: Dict[str, Any] = {
            "tournaments": [],
            "seasons": {},
            "cuptrees": {}
        }

        tournaments = self.get_tournaments(save_dir=save_dir)
        if max_tournaments is not None:
            tournaments = tournaments[:max_tournaments]
        all_data["tournaments"] = tournaments

        for t in tournaments:
            tid = t["id"]
            try:
                seasons = self.get_seasons(tid, save_dir=save_dir)
                all_data["seasons"][tid] = seasons
                # time.sleep(1)  # Add delay between tournament requests
                for season in seasons:
                    sid = season["id"]
                    try:
                        cuptrees = self.get_cuptrees(tid, sid, save_dir=save_dir)
                        all_data["cuptrees"][(tid, sid)] = cuptrees
                    except Exception as e:
                        logging.warning(f"Failed to get cuptrees for tournament {tid} season {sid}: {e}")
                    # time.sleep(1)  # Add delay between season requests
            except Exception as e:
                logging.warning(f"Failed to get seasons for tournament {tid}: {e}")

        return all_data
    
    def close(self):
        """
        Closes the Selenium WebDriver.
        """
        if self.driver:
            self.driver.quit()

    def __exit__(self):
        """
        Ensures the WebDriver is closed when exiting the context.
        """
        self.close()

if __name__ == "__main__":
    fetcher = TennisDataFetcher()
    data = fetcher.get_all_data(max_tournaments=1)
    fetcher.close()
