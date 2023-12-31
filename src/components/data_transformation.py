import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    def adjust_col_vals(self):
        epoch = pd.Timestamp("2015-01-01", tz="UTC")

        self.data["kickoff_time"] = (
            (pd.to_datetime(self.data["kickoff_time"]) - epoch)
            .astype("timedelta64[s]")
            .astype(int)
        )

        int_cols = [
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
            "yellow_cards",
            "starts",
        ]  # list of integer columns

        object_cols = ["was_home"]  # list of object/string columns

        imputer = SimpleImputer(strategy="median")
        self.data = pd.DataFrame(
            imputer.fit_transform(self.data), columns=self.data.columns
        )

        # Convert data types
        self.data[int_cols] = self.data[int_cols].astype(int)
        self.data[object_cols] = self.data[object_cols].astype(object)

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

    if os.path.exists(output_csv):
        logging.info(f"Deleting tmp {output_csv}")
        os.remove(output_csv)
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                try:
                    filepath = os.path.join(directory, filename)
                    transformer = DataTransformer(filepath, overlap)
                    transformer.keep_cols()
                    transformer.adjust_col_vals()
                    transformer.merge_dataframes_side_by_side()
                    transformer.append_df_to_csv(output_csv)
                except Exception as e:
                    logging.error(f"Error in transformation: {e}")

    except Exception as e:
        raise CustomException(e)


class DataProcessor:
    def __init__(self, filename, target_col):
        self.df = pd.read_csv(filename)  # .iloc[0:1000]
        self.target_col = target_col

    def process(self):
        X = self.df.drop(self.target_col, axis=1)

        cols_to_drop = [col for col in X.columns if col.endswith("_2")]
        X = X.drop(cols_to_drop, axis=1)

        y = self.df[self.target_col]

        logging.info(f"Splitting into numerical and catagorical")
        X_cat = X.select_dtypes(include=["object"])
        X_num = X.select_dtypes(exclude=["object"])

        logging.info(f"Processing")
        X_cat_processed = CategoricalProcessor().process(X_cat)
        X_num_processed = NumericalProcessor().process(X_num)

        logging.info(f"Merging")
        X_processed = pd.concat([X_cat_processed, X_num_processed], axis=1)

        return X_processed, y


class CategoricalProcessor:
    def process(self, X_cat):
        encoder = OneHotEncoder(sparse_output=False)
        X_cat_encoded = encoder.fit_transform(X_cat)
        return pd.DataFrame(
            X_cat_encoded, columns=encoder.get_feature_names_out(X_cat.columns)
        )


class NumericalProcessor:
    def process(self, X_num):
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        return pd.DataFrame(X_num_scaled, columns=X_num.columns)
