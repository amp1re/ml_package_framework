import pandas as pd

from .base import DataProcessor


class SberDataProcessor(DataProcessor):
    """
    A subclass of DataProcessor for handling data specific to Sber.

    This class extends the functionality of the DataProcessor class, providing
    implementations for data loading and processing that are tailored to the
    specific requirements and formats of data associated with Sber.

    Methods inherited from DataProcessor are `load_data` and `process_data`,
    which should be overridden in this subclass to handle Sber-specific data formats.

    See Also
    --------
    DataProcessor : The base class from which this class inherits.

    Notes
    -----
    The actual implementation details should be provided in the methods of this class.
    It's assumed that the data formats and processing requirements are specific to Sber
    and may differ from general data processing methods.

    Examples
    --------
    >>> sber_processor = SberDataProcessor()
    >>> df = sber_processor.load_data('sber_data.csv')
    >>> processed_df = sber_processor.process_data(df)
    """

    def load_data(self, path: str, *args, **kwargs) -> pd.DataFrame:
        """
        Load data from a specified file path into a pandas DataFrame.

        This method supports loading data from files with extensions 'csv' and 'json'.
        It determines the file type based on the file extension and uses the appropriate
        pandas function to load the data.

        Parameters
        ----------
        path : str
            The file path from which to load the data. Supported file types are 'csv' and 'json'.
        *args
            Variable length argument list passed to the pandas reading function.
        **kwargs
            Arbitrary keyword arguments passed to the pandas reading function.

        Returns
        -------
        pd.DataFrame
            The loaded data as a pandas DataFrame.

        Raises
        ------
        ValueError
            If the file extension is not supported (not 'csv' or 'json').

        Examples
        --------
        >>> processor = DataProcessor()
        >>> df = processor.load_data('data.csv')
        >>> df_json = processor.load_data('data.json')
        """

        file_extension = path.split(".")[-1]
        if file_extension == "csv":
            loaded_data = pd.read_csv(path, *args, **kwargs)
        elif file_extension == "json":
            loaded_data = pd.read_json(path, *args, **kwargs)
        else:
            raise ValueError(
                f"File with the extension '{file_extension}' is not supported."
            )

        return loaded_data

    def process_data(self, *args, **kwargs):
        """
        Process the loaded data.
        """
