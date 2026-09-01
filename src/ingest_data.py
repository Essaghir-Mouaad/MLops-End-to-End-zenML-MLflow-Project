import os
import zipfile
from abc import ABC,abstractmethod
import pandas as pd 


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest data from the data source
        """
        pass


class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Extract the data from zip file
        """
        if not file_path.endswith('.zip'):
            raise ValueError('File path is not a zip file')

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('extract_path')

        extracted_files = os.listdir('extract_path')
        csv_files = [f for f in extracted_files if f.endswith('.csv')]
        if len(csv_files) == 0:
            raise ValueError('No csv files found in the extracted files')
        if len(csv_files) > 1:
            raise ValueError('More than one csv file found in the extracted files')

        csv_file_path = os.path.join('extract_path', csv_files[0])
        return pd.read_csv(csv_file_path) 


# implement the Factory to create a data ingestor
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == '.zip':
            return ZipDataIngestor()
        else:
            raise ValueError('Unsupported file type')   


# Example usage:
if __name__ == "__main__":
    # # Specify the file path
    # file_path = "/home/mouaad/Desktop/Projects/Learn/MLops/house_price_prediction/data/archive.zip"

    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[1]

    # # Get the appropriate DataIngestor
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # # Ingest the data and load it into a DataFrame
    # df = data_ingestor.ingest(file_path)

    # # Now df contains the DataFrame from the extracted CSV
    # print(df.head())  # Display the first few rows of the DataFrame
    pass