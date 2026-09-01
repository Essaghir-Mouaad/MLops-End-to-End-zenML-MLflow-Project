from abc import ABC, abstractmethod
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.generate_correlation_matrix(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_matrix(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass


class SimpleMultivariateAnalyzer(MultivariateAnalysisTemplate):
    def generate_correlation_matrix(self, df: pd.DataFrame):
        print("\n Generating Correlation Matrix")
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f')
        plt.title("Correlation Matrix of Features")
        plt.show()


    def generate_pairplot(self, df: pd.DataFrame):
        print("\n Generating Pairplot")
        
        sns.pairplot(df)
        plt.suptitle("Pairplot of Features", y=1.02)   
        plt.show()