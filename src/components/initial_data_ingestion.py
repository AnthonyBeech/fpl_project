import pandas as pd
import requests
import json
import time
import logging
import os

from src.exception import CustomException
from src.logger import logging


class PlayerDataProcessor:
    """
    Handles processing and updating of player data.
    """

    def __init__(self, base_url: str, data_dir: str, sum_dir: str):
        """
        Initializes the data processor with URLs and directories.
        """
        self.base_url = base_url
        self.data_dir = data_dir
        self.sum_dir = sum_dir
        os.makedirs(self.sum_dir, exist_ok=True)

    def get_individual_player_data(self, player_id: int) -> dict:
        """
        Retrieves player-specific detailed data.
        """
        full_url = self.base_url + f"element-summary/{player_id}/"
        response = ""
        while response == "":
            try:
                response = requests.get(full_url)
            except:
                time.sleep(5)
        if response.status_code != 200:
            logging.error(f"Response was code {response.status_code}")
            return None
        return json.loads(response.text)

    def get_data(self) -> dict:
        """
        Retrieves FPL player data.
        """
        response = requests.get(self.base_url + "bootstrap-static/")
        if response.status_code != 200:
            logging.error(f"Response was code {response.status_code}")
            return None
        return json.loads(response.text)

    def process_player_data(self):
        """
        Processes and updates player data.
        """
        try:
            sdata = self.get_data()
            if sdata is None:
                raise CustomException("Failed to get initial data")

            for ID in range(1, len(sdata["elements"])):
                logging.info(f"Processing ID: {ID}")
                fnm = sdata["elements"][ID]["first_name"]
                lnm = sdata["elements"][ID]["second_name"]
                nm = f"{fnm}_{lnm}.csv"
                try:
                    df = pd.read_csv(f"{self.data_dir}/{nm}")
                except:
                    logging.error(f"File does not exist: {nm}")
                    continue
                    
                most_recent_kickoff = df["kickoff_time"].max()

                edata = self.get_individual_player_data(ID)
                if edata is None:
                    continue

                edata_r = list(reversed(edata["history"]))
                for i, data in enumerate(edata_r):
                    if data["kickoff_time"] == most_recent_kickoff:
                        break

                dfr = pd.DataFrame(edata_r[:i])
                result_df = pd.concat([df, dfr], ignore_index=True)
                result_df.to_csv(f"{self.sum_dir}/{nm}")

        except Exception as e:
            logging.error(f"Error in processing player data: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    processor = PlayerDataProcessor(
        "https://fantasy.premierleague.com/api/", "data/legacy/combined", "data/sum"
    )
    processor.process_player_data()
