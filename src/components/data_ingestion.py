import requests
import json
import os
import shutil
import re
import glob
from collections import defaultdict
import pandas as pd
import time

from src.exception import CustomException
from src.logger import logging


class CsvExtractor:
    """
    Handles extraction, processing, and combination of CSV files.
    """

    def __init__(
        self,
        years: list,
        extracted_dir: str,
        combined_dir: str,
        file_pattern: str = "*/gw.csv",
    ):
        """
        Initializes the CsvExtractor with directories and file pattern.
        """
        self.years = years
        self.extracted_dir = extracted_dir
        self.combined_dir = combined_dir
        self.file_pattern = file_pattern
        os.makedirs(self.extracted_dir, exist_ok=True)
        os.makedirs(self.combined_dir, exist_ok=True)

    def extract_copy_csv(self, year: int, source_dir: str):
        """
        Extracts and copies CSV files for a given year from source_dir to extracted_dir.
        """
        try:
            logging.info(f"Processing year {year} started")

            players = set()
            search_str = os.path.join(source_dir, self.file_pattern)

            for filepath in glob.glob(search_str):
                player_name_with_tag = os.path.basename(
                    os.path.dirname(filepath)
                )  # fname_lname_034
                player_name = re.sub(r"_\d+", "", player_name_with_tag)  # fname_lname

                if player_name not in players:
                    players.add(player_name)
                    new_filename = f"{player_name}_20{year}.csv"
                    shutil.copy(
                        filepath, os.path.join(self.extracted_dir, new_filename)
                    )
                else:
                    logging.warning(f"Duplicate player: {player_name}")

            logging.info(f"Processing year {year} completed.")

        except Exception as e:
            logging.error(f"Error processing year {year}: {e}")
            raise CustomException(e)

    def group_and_combine_files(self):
        """
        Groups files by player name and combines them into single CSV files.
        """
        logging.info(f"Grouping files together in {self.extracted_dir}")

        try:
            file_groups = defaultdict(list)

            for file_path in glob.glob(os.path.join(self.extracted_dir, "*.csv")):
                base_name, year = self._parse_file_name(file_path)
                file_groups[base_name].append((year, file_path))

            for name, files in file_groups.items():
                combined_df = self._combine_files(files)
                combined_df.to_csv(
                    os.path.join(self.combined_dir, f"{name}.csv"), index=False
                )

            logging.info("All files combined and written to CSV")

        except Exception as e:
            logging.error(f"Error combining files: {e}")
            raise CustomException(e)

    def _parse_file_name(self, file_path: str) -> tuple:
        """
        Parses file name to extract base name and year.
        """
        parts = os.path.basename(file_path).split("_")
        base_name = "_".join(parts[:-1])
        year = parts[-1].split(".")[0]
        return base_name, year

    def _combine_files(self, files: list) -> pd.DataFrame:
        """
        Combines list of file paths into a single DataFrame.
        """
        sorted_files = sorted(files, key=lambda x: x[0])
        df_list = [pd.read_csv(file[1]) for file in sorted_files]
        return pd.concat(df_list, ignore_index=True)


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

    def process_player_data(self):
        """
        Processes and updates player data.
        """

        sdata = self._get_data()

        if sdata is None:
            raise CustomException("Failed to get data from _getdata()")

        for ID in range(1, len(sdata["elements"])):
            logging.info(f"Processing ID: {ID}")

            fnm = sdata["elements"][ID]["first_name"]
            lnm = sdata["elements"][ID]["second_name"]
            nm = f"{fnm}_{lnm}.csv"
            position = sdata["element_types"][sdata["elements"][ID]["element_type"] - 1]["id"]

            try:
                df = pd.read_csv(f"{self.data_dir}/{nm}")
            except:
                logging.error(f"File does not exist: {nm}")
                continue
            
            try:
                df = df.drop(['position'], axis=1)
            except:
                logging.warning(f"No position data to remove")
                

            most_recent_kickoff = df["kickoff_time"].max()

            edata = self._get_individual_player_data(ID)
            if edata is None:
                logging.error(f"No data at player {ID}")
                continue

            # find any data not in latest
            edata_r = list(reversed(edata["history"]))
            for i, data in enumerate(edata_r):
                if data["kickoff_time"] == most_recent_kickoff:
                    break

            dfr = pd.DataFrame(edata_r[:i])
            result_df = pd.concat([df, dfr], ignore_index=True)
            result_df["position"] = position
            result_df.to_csv(f"{self.sum_dir}/{nm}")
            

    def _get_data(self) -> dict:
        """
        Retrieves FPL player data.
        """
        response = requests.get(self.base_url + "bootstrap-static/")
        if response.status_code != 200:
            logging.error(f"Response was code {response.status_code}")
            return None
        return json.loads(response.text)

    def _get_individual_player_data(self, player_id: int) -> dict:
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

    def rename(self):
        """
        Renames the folder to data/latest
        """
        logging.info(f"Renaming to {self.data_dir}")
        try:
            if os.path.exists(self.data_dir):
                logging.info(f"Deleting tmp {self.sum_dir}")
                shutil.rmtree(self.data_dir)
            os.rename(self.sum_dir, self.data_dir)

        except Exception as e:
            raise CustomException(e)

        logging.info(f"Renamed to {self.data_dir}")


class CheckApiForChanges:
    """
    Manages data retrieval and comparison for Fantasy Premier League.
    """

    def __init__(self, api_url: str):
        """
        Initializes the data handler.
        """
        self.api_url = api_url
        self.old_data_file = "data/previous_api_response.json"

    def is_changed(self) -> bool:
        """
        Checks for API changes and returns True if changes are detected.
        """
        self.new_data = self._fetch_data()
        return self._is_data_changed()

    def _fetch_data(self) -> dict:
        """
        Fetches data from the FPL API.
        """
        response = requests.get(f"{self.api_url}bootstrap-static/")
        logging.info(f"Got response from API")
        return json.loads(response.text)["events"]

    def _is_data_changed(self) -> bool:
        """
        Determines if new data differs from previous data.
        """
        try:
            with open(self.old_data_file, "r") as file:
                old_data = json.load(file)
                logging.info(f"Old API request found")

        except FileNotFoundError:
            logging.warning(f"No old API request found")
            old_data = None

        if old_data != self.new_data:
            logging.info(f"APIs different")

            with open(self.old_data_file, "w") as file:
                json.dump(self.new_data, file)
            return True

        logging.info(f"APIs the same, no changes needed")
        return False
