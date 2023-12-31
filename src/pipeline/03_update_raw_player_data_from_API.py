from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import CheckApiForChanges, PlayerDataProcessor


@dataclass
class Config:
    base_url: str = "https://fantasy.premierleague.com/api/"
    data_dir: str = "data/latest" 
    sun_dir: str = "data/sum"


cfg = Config

if __name__ == "__main__":
    logging.info(f"Checking API for chaneges")

    api_checker = CheckApiForChanges(cfg.base_url)

    if api_checker.is_changed():
        logging.info(f"Adding changes to latest dataset")

        data_processor = PlayerDataProcessor(cfg.base_url, cfg.comb_dir, cfg.lat_dir)
        try:
            data_processor.process_player_data()
            data_processor.rename()
        except Exception as e:
            raise CustomException(e)

        logging.info(f"Changes added to latest dataset")

    else:
        logging.info(f"No change to API request. Try again in 24hrs")
