import os
import pandas as pd
import shutil
import glob
import re
from collections import defaultdict

from src.logger import logging
from src.exception import CustomException


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
            for filepath in glob.glob(os.path.join(source_dir, self.file_pattern)):
                player_name_with_tag = os.path.basename(os.path.dirname(filepath))
                player_name = re.sub(r"_\d+", "", player_name_with_tag)

                if player_name not in players:
                    players.add(player_name)
                    new_filename = f"{player_name}_20{year}.csv"
                    shutil.copy(
                        filepath, os.path.join(self.extracted_dir, new_filename)
                    )

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


if __name__ == "__main__":
    logging.info("Current Working Directory: {}".format(os.getcwd()))
    years = list(range(16, 24))
    extractor = CsvExtractor(years, "data/legacy/extracted/", "data/legacy/combined/")

    for year in years:
        source_dir = f"data/legacy/20{year}-{year+1}/players/"
        extractor.extract_copy_csv(year, source_dir)

    extractor.group_and_combine_files()
