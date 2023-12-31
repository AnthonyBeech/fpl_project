import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

class DataProcessor:
    def __init__(self, filename, target_col):
        self.df = pd.read_csv(filename).iloc[0:100]
        self.target_col = target_col
        

    def process(self):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]
        
        logging.info(f"Splitting into numerical and catagorical")
        X_cat = X.select_dtypes(include=['object'])
        X_num = X.select_dtypes(exclude=['object'])

        logging.info(f"Processing")
        X_cat_processed = CategoricalProcessor().process(X_cat)
        X_num_processed = NumericalProcessor().process(X_num)

        logging.info(f"Merging")
        X_processed = pd.concat([X_cat_processed, X_num_processed], axis=1)

        return X_processed, y

class CategoricalProcessor:
    def process(self, X_cat):
        encoder = OneHotEncoder(sparse_output=False)
        X_cat_encoded = encoder.fit_transform(X_cat)
        return pd.DataFrame(X_cat_encoded, columns=encoder.get_feature_names_out(X_cat.columns))

class NumericalProcessor:
    def process(self, X_num):
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        return pd.DataFrame(X_num_scaled, columns=X_num.columns)


if __name__=="__main__":
    data_processor = DataProcessor('data/raw.csv', target_col='bps_2')
    X, y = data_processor.process()

    logging.info(f"Splitting into test and train")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    logging.info(f"Saving to csv")
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
