from IPython.core.pylabtools import figsize
from abc import ABC, abstractmethod
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns   


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature_1: str, feature_2: str):
        pass


class NumericalBivariateAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature_1: str, feature_2: str):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature_1, y=feature_2, data=df)
        plt.title(f'Relationship between {feature_1} and {feature_2}')
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.show()

    
class CategoricalBivariateAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature_1: str, feature_2: str):
        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature_1, y=feature_2, data=df)
        plt.title(f'Relationship between {feature_1} and {feature_2}')
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.show()


class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self.strategy = strategy
    
    def execute_analysis(self, df: pd.DataFrame, feature_1: str, feature_2: str):
        self.strategy.analyze(df, feature_1, feature_2)

    