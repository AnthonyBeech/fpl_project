from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import process_all_csv_files


@dataclass
class Config:
    latest_dir: str = "data/latest/"
    raw_data_dir: str = "data/raw.csv"
    overlap: int = 3


cfg = Config

if __name__ == "__main__":
    logging.info("transforming data into new structure ready for splitting")
    
    try:
        process_all_csv_files(cfg.latest_dir, cfg.raw_data_dir, cfg.overlap)
    except Exception as e:
            raise CustomException(e)
        
    logging.info(f"All files written to csv")
