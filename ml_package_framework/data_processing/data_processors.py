import pandas as pd

from .base import DataProcessor


class SberDataProcessor(DataProcessor):
    """
    SberDataProcessor is a subclass of DataProcessor for handling data specifically from Sber.
    """

    def load_data(self, path: str, *args, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file located at the given path.

        :param path: A string representing the file path to the CSV file.
        :return: A pandas DataFrame containing the loaded data.
        """
        return pd.read_csv(path)

    def process_data(self, *args, **kwargs):
        """
        Process the loaded data.
        """
