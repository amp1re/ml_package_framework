from abc import ABC, abstractmethod


class DataProcessor(ABC):
    """
    Base class for data processing.
    This class provides a template for loading and processing data.
    Subclasses should implement the load_data and process_data methods.
    """

    @abstractmethod
    def load_data(self, path: str, *args, **kwargs):
        """
        Load data from a source.

        :param path: A string representing the file path.
        :param args: Positional arguments passed to the data loading method.
        :param kwargs: Keyword arguments passed to the data loading method.
        :return: Loaded data.
        """
        raise NotImplementedError(
            "The load_data method must be implemented by subclasses."
        )

    @abstractmethod
    def process_data(self, data, *args, **kwargs):
        """
        Process the loaded data.

        :param data: Data to be processed.
        :param args: Positional arguments passed to the data processing method.
        :param kwargs: Keyword arguments passed to the data processing method.
        :return: Processed data.
        """
        raise NotImplementedError(
            "The process_data method must be implemented by subclasses."
        )
