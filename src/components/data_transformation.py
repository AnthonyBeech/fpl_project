import pandas as pd
import os

from src.exception import CustomException
from src.logger import logging


class DataTransformer:
    """
    Transforms data into overlapped sequences for deep learning models.
    """

    def __init__(self, filepath: str, overlap: int):
        """
        Initializes the preprocessor with the file path.
        """
        self.filepath = filepath
        self.overlap = overlap
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from a CSV file.
        """
        logging.info(f"Loading: {self.filepath}")

        return pd.read_csv(self.filepath)

    def keep_cols(self) -> None:
        """
        Keep relevant columns.
        """
        keep_cols = [
            "assists",
            "bonus",
            "bps",
            "clean_sheets",
            "creativity",
            "goals_conceded",
            "goals_scored",
            "ict_index",
            "influence",
            "kickoff_time",
            "minutes",
            "own_goals",
            "penalties_missed",
            "penalties_saved",
            "red_cards",
            "saves",
            "team_a_score",
            "team_h_score",
            "threat",
            "value",
            "was_home",
            "yellow_cards",
            "starts",
        ]
        self.data = self.data[keep_cols]

    def merge_dataframes_side_by_side(self) -> None:
        """
        Merges DataFrame rows side by side, appending suffixes for overlapping columns.
        """
        dfs = pd.DataFrame()
        for i in range(len(self.data) - self.overlap + 1):
            df1 = self.data.iloc[[i]]
            for j in range(1, self.overlap):
                df2 = self.data.iloc[[i + j]]
                df2_renamed = df2.rename(columns=lambda x: f"{x}_{j}")
                df1 = pd.concat(
                    [df1.reset_index(drop=True), df2_renamed.reset_index(drop=True)],
                    axis=1,
                )

            dfs = pd.concat([dfs, df1], ignore_index=True)

        self.data = dfs

    def append_df_to_csv(self, csv_file_path: str) -> None:
        """
        Appends the DataFrame to a CSV file, creating the file if it doesn't exist.
        """
        logging.info(f"Writing {csv_file_path} to csv")

        file_exists = os.path.isfile(csv_file_path)
        self.data.to_csv(csv_file_path, mode="a", header=not file_exists, index=False)


def process_all_csv_files(directory: str, output_csv: str, overlap: int) -> None:
    """
    Processes all CSV files in the given directory.
    """
    logging.info(f"Processing files in {directory}")

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            try:
                filepath = os.path.join(directory, filename)
                transformer = DataTransformer(filepath, overlap)
                transformer.keep_cols()
                transformer.merge_dataframes_side_by_side()
                transformer.append_df_to_csv(output_csv)
            except Exception as e:
                logging.error(f"Error in transformation: {e}")


if __name__ == "__main__":
    process_all_csv_files("data/latest/", "data/raw.csv", overlap=3)
    
    logging.info(f"All files written to csv")
