from abc import ABC, abstractmethod
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        pass


class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        if not feature in df.columns:
            raise ValueError(f'Feature {feature} not found in the DataFrame')

        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(f'Feature {feature} is not numeric')

        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], kde=False)
        plt.title(f'Distribution of {feature}')
        
        plt.xlabel(feature)

        plt.ylabel('Frequency')

        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10,6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f'Distribution of {feature}')
        
        plt.xlabel(feature)

        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

        

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self.strategy = strategy
    
    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self.strategy.analyze(df, feature)