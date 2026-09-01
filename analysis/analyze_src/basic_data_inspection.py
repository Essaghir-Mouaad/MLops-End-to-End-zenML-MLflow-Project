from abc import ABC, abstractmethod
import pandas as pd

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, data: pd.DataFrame) -> pd.DataFrame:
        pass    



class DataTypeInspectorStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(data.info())


        
    

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def execute_strategy(self, data: pd.DataFrame):
        return self._strategy.inspect(data)




# Example usage
if __name__ == "__main__":
    # # Example usage of the DataInspector with different strategies.

    # # Load the data
    # df = pd.read_csv('/home/mouaad/Desktop/Projects/Learn/MLops/house_price_prediction/extract_path/AmesHousing.csv')

    # # Initialize the Data Inspector with a specific strategy
    # inspector = DataInspector(DataTypeInspectorStrategy())
    # inspector.execute_strategy(df)

    # # Change strategy to Summary Statistics and execute
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_strategy(df)
    pass