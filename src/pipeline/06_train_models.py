from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.model_training import ModelTrainer


@dataclass
class Config:
    data_dir: str = "data/"
    model_dir: str = "data/model.pkl"
    plot_dir: str = "plot/"
    samples: int = 1000


cfg = Config


if __name__ == "__main__":
    logging.info("Training model.")

    model_trainer = ModelTrainer(cfg.data_dir, cfg.model_dir, cfg.plot_dir)
    try:
        model_trainer.initiate_model_data(cfg.samples)
        model_trainer.initiate_model_params()
        report = model_trainer.train_model()
        logging.info(report)
        logging.info("FInished.")

    except Exception as e:
        raise CustomException(e)
