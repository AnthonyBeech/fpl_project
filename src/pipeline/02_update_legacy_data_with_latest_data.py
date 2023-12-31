from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import PlayerDataProcessor


@dataclass
class Config:
    base_url: str = "https://fantasy.premierleague.com/api/"
    combined_dir: str = "data/legacy/combined"
    latest_dir: str = "data/latest"


cfg = Config

if __name__ == "__main__":
    logging.info("Updateing legacy with latest data to make backup")

    processor = PlayerDataProcessor(cfg.base_url, cfg.combined_dir, cfg.latest_dir)

    try:
        processor.process_player_data()

    except Exception as e:
        raise CustomException(e)
