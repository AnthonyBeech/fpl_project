from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataProcessor


@dataclass
class Config:
    raw_data_dir: str = "data/raw.csv"
    target_col: str = "bps_2"
    X_train_dir: str = "data/X_train.csv"
    X_test_dir: str = "data/X_test.csv"
    y_train_dir: str = "data/y_train.csv"
    y_test_dir: str = "data/y_test.csv"
    test_size: int = 0.1


cfg = Config


if __name__ == "__main__":
    logging.info("Final data processing")
    
    data_processor = DataProcessor(cfg.raw_data_dir, cfg.target_col)
    
    try:
        X, y = data_processor.process()
    except Exception as e:
            raise CustomException(e)

    logging.info(f"Splitting into test and train")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=42
    )

    logging.info(f"Saving to csv")
    
    X_train.to_csv(cfg.X_train_dir, index=False)
    X_test.to_csv(cfg.X_test_dir, index=False)
    y_train.to_csv(cfg.y_train_dir, index=False)
    y_test.to_csv(cfg.y_test_dir, index=False)
