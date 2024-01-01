import os
import sys

import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve




def save_plot(plot, file_name):
    plot_dir = "plots/"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, file_name)
    plot.savefig(plot_path)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Testing model: {model}")

            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            logging.info(f"Predicting with model: {best_model}")
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Plotting Predicted vs True values
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_test_pred)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Predicted vs True Values - {model_name}')

            # Add a line where x equals y
            plt.plot(y_test, y_test, color='red', linestyle='--')  # Add this line

            # Save the plot
            save_plot(plt, f"predicted_vs_true_{model_name}.png")

            # Variance-Bias Analysis
            # [Optional: Code to plot variance-bias analysis]

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            logging.info(f"Adding model to report: {model_name}")
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

