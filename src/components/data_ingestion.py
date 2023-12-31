import requests
import json

from initial_data_ingestion import PlayerDataProcessor
from src.exception import CustomException
from src.logger import logging


class CheckApiForChanges:
    """
    Manages data retrieval and comparison for Fantasy Premier League.
    """

    def __init__(self, api_url: str):
        """
        Initializes the data handler.
        """
        self.api_url = api_url

    def fetch_data(self) -> dict:
        """
        Fetches data from the FPL API.
        """
        response = requests.get(f"{self.api_url}bootstrap-static/")
        logging.info(f"Got response from API")
        return json.loads(response.text)["events"]

    def is_data_changed(
        self, new_data: dict, old_data_file: str = "data/previous_api_response.json"
    ) -> bool:
        """
        Determines if new data differs from previous data.
        """
        try:
            with open(old_data_file, "r") as file:
                old_data = json.load(file)
                logging.info(f"Old API request found")
                
        except FileNotFoundError:
            logging.warning(f"No old API request found")
            old_data = None

        if old_data != new_data:
            logging.info(f"APIs different")
            
            with open(old_data_file, "w") as file:
                json.dump(new_data, file)
            return True
        
        logging.info(f"APIs the same, no changes needed")
        return False

    def is_changed(self) -> bool:
        """
        Checks for API changes and returns True if changes are detected.
        """
        new_data = self.fetch_data()
        return self.is_data_changed(new_data)


if __name__ == "__main__":
    try:
        api_checker = CheckApiForChanges("https://fantasy.premierleague.com/api/")

        if api_checker.is_changed():
            logging.info(f"Adding changes to latest dataset")
            
            data_processor = PlayerDataProcessor(
                "https://fantasy.premierleague.com/api/", "data/sum", "data/latest"
            )
            data_processor.process_player_data()
            
            logging.info(f"Changes added to latest dataset")
            
        else:
            logging.info(f"No change to API request. Try again in 24hrs")

    except Exception as e:
        logging.error(f"Error in processing player data: {e}")
        raise CustomException(e)
