import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


# setup the logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



class DataSplitterStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, traget_columns: str):
        pass


class SimpletrainTestSplitStrategy(DataSplitterStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, traget_columns: str):
        
        logging.info("Spliting the data into training and testing sets .....")
        
        X = df.drop(columns=[traget_columns])
        y = df[traget_columns]


        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        logging.info("Data split successfully .... ")

        return X_train, X_test, y_train, y_test 


class DataSplittingContext:
    def __init__(self, strategy: DataSplitterStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: DataSplitterStrategy):
        logging.info("Setting new strategy .... ")
        self.strategy = strategy
        logging.info("Strategy changed successfully .....")

    def split(self, df: pd.DataFrame, traget_columns: str):
        logging.info("Splitting data .... ")
        return self.strategy.split_data(df, traget_columns)   