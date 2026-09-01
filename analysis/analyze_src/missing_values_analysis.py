from abc import ABC, abstractmethod
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 


# defining the template for the missing values

class MissingValuesAnalysisTemplate(ABC):

    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)

        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass 

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass


class SimpleMissingValuesAnalyzer(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        print("\n=== Identifying Missing Values ===")
        missing_values = df.isna().sum()

        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):

        print("\n=== Visualizing Missing Values ===")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isna(), cbar=True, cmap='inferno')
        plt.title('Missing Values Visualization')
        plt.show()

    