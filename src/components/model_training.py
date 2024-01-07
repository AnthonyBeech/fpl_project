import pandas as pd
import shutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging


class ModelTrainer:
    def __init__(self, data_loc, model_loc, plot_loc):
        self.data_loc = data_loc
        self.model_loc = model_loc
        self.plot_dir = plot_loc
        self.evaluator = ModelEvaluater(data_loc, model_loc, plot_loc)

        self._delete_old_plots()

    def initiate_model_data(self, samples):
        logging.info("Split training and test input data")

        self.X_train = pd.read_csv(self.data_loc + "X_train.csv").values[:samples]
        self.y_train = pd.read_csv(self.data_loc + "y_train.csv").values.ravel()[:samples]
        self.X_test = pd.read_csv(self.data_loc + "X_test.csv").values[:samples//10]
        self.y_test = pd.read_csv(self.data_loc + "y_test.csv").values.ravel()[:samples//10]

        logging.info(
            f"Model shapes are: {self.X_train.shape}, {self.X_test.shape}, {self.y_train.shape}, {self.y_test.shape}"
        )

    def initiate_model_params(self):
        logging.info("Setting up models and params")

        # self.models = {
        #     "Random Forest": RandomForestRegressor(),
        #     "Linear Regression": LinearRegression(),
        # }

        # self.params = {
        #     "Random Forest": {"n_estimators": [128]},
        #     "Linear Regression": {},
        # }
        
        self.models = {
            "Linear Regression": LinearRegression(),
        }

        self.params = {
            "Linear Regression": {},
        }

    def train_model(self):
        logging.info("Evauating models")

        self.model_report: dict = self.evaluator.evaluate_models(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            models=self.models,
            param=self.params,
        )

        return self.model_report

    def _delete_old_plots(self):
        if os.path.exists(self.plot_dir):
            logging.info(f"Deleting tmp plots")

            shutil.rmtree(self.plot_dir)


class ModelEvaluater:
    def __init__(self, data_loc, model_loc, plot_loc):
        self.data_loc = data_loc
        self.model_loc = model_loc
        self.plot_dir = plot_loc

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, param):
        report = {}

        for model_name, model in models.items():
            logging.info(f"Testing model: {model}")

            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)
            
            logging.info(pd.DataFrame(gs.cv_results_))

            logging.info(f"Predicting with model: {best_model}")

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            self._plot_model_predicted(y_test, y_test_pred, model_name)
            self._plot_model_predicted(y_train, y_train_pred, model_name + "_train")


            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"Adding model to report: {model_name}")

            report[model_name] = train_model_score, test_model_score

        return report

    def _plot_model_predicted(self, y, y_pred, model_name):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y, y=y_pred)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.ylim(0, 10)
        plt.title(f"Predicted vs True Values - {model_name}")
        
        # Calculate the coefficients (slope and intercept) for the line of best fit
        slope, intercept = np.polyfit(y, y_pred, 1)
        
        # Plot the line of best fit
        plt.plot(y, slope * y + intercept, color="red", linestyle="--", label="Best Fit Line")
        
        # Plot the ideal fit line (slope=1, intercept=0)
        plt.plot(y, y, color="green", linestyle="--", label="Ideal Fit Line")
        
        plt.legend()  # Add a legend
        
        self._save_plot(plt, f"predicted_vs_true_{model_name}.png")

    def _save_plot(self, plot, file_name):
        os.makedirs(self.plot_dir, exist_ok=True)

        plot_path = os.path.join(self.plot_dir, file_name)
        plot.savefig(plot_path)
