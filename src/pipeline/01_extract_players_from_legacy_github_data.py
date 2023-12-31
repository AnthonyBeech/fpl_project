import os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import CsvExtractor


@dataclass
class Config:
    year_min: int = 16
    year_max: int = 24
    extracted_dir: str = "data/legacy/extracted/"  # Save csv files ectracted from legact data
    combined_dir: str = "data/legacy/combined/" # Save csv files are combining by name


cfg = Config

if __name__ == "__main__":
    logging.info("Extracting playeres from legacy github")

    years = list(range(cfg.year_min, cfg.year_max))

    extractor = CsvExtractor(
        years,
        extracted_dir=cfg.extracted_dir,
        combined_dir=cfg.combined_dir,
    )

    for year in years:
        try:
            source_dir = f"data/legacy/20{year}-{year+1}/players/"
            extractor.extract_copy_csv(year, source_dir)
        except Exception as e:
            logging.error(e)

    try:
        extractor.group_and_combine_files()
    except Exception as e:
        raise CustomException(e)
