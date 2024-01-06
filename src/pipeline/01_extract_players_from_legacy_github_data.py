

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import CsvExtractor


@dataclass
class Config:
    year_min: int = 16
    year_max: int = 24
    extracted_dir: str = "data/legacy/extracted/"  # location to save extracted gw data
    combined_dir: str = "data/legacy/combined/" # location to save combined gw data
    source_dir: str = "data/legacy/20yy_yy/players/"  # location to read legacy data




cfg = Config

if __name__ == "__main__":
    logging.info(f"Extracting gw data from leagacy github data in {cfg.source_dir} to {cfg.extracted_dir}")

    years = list(range(cfg.year_min, cfg.year_max))

    extractor = CsvExtractor(
        years,
        extracted_dir=cfg.extracted_dir,
        combined_dir=cfg.combined_dir,
    )

    for year in years:
        try:
            source_dir = cfg.source_dir.replace("yy_yy", f"{year}-{year+1}")
            extractor.extract_copy_csv(year, source_dir)
            
        except Exception as e:
            logging.error(e)

    try:
        logging.info(f"grouping files from {cfg.extracted_dir} to {cfg.combined_dir}")
        extractor.group_and_combine_files()
        
    except Exception as e:
        logging.error(e)
        raise CustomException(e)
