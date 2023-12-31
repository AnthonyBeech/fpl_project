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



    def save_object(file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)

            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

        except Exception as e:
            raise CustomException(e, sys)


    def evaluate_models(X_train, y_train, X_test, y_test, models, param):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                logging.info(f"Testing model: {model}")
                
                para = param[list(models.keys())[i]]

                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                logging.info(f"Predicting with model: {model}")
                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)

                test_model_score = r2_score(y_test, y_test_pred)
                logging.info(f"Adding model to report: {model}")
                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)
