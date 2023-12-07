from abc import ABC, abstractmethod


class DataProcessor(ABC):
    """
    Base class for data processing.

    This class provides a template for subclasses to implement methods for loading
    and processing data. It defines abstract methods `load_data` and `process_data`
    which must be implemented in any subclass.

    Methods
    -------
    load_data(path, *args, **kwargs)
        Abstract method to load data from a specified path.
    process_data(data, *args, **kwargs)
        Abstract method to process the loaded data.
    """

    @abstractmethod
    def load_data(self, path: str, *args, **kwargs):
        """
        Load data from a source.

        This is an abstract method that should be implemented by subclasses to
        load data from the given path.

        Parameters
        ----------
        path : str
            The file path or data source from which to load data.
        *args
            Variable length argument list for additional parameters.
        **kwargs
            Arbitrary keyword arguments for additional parameters.

        Returns
        -------
        The loaded data, the format of which will depend on the implementation
        in the subclass.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "The load_data method must be implemented by subclasses."
        )

    @abstractmethod
    def process_data(self, data, *args, **kwargs):
        """
        Process the loaded data.

        This is an abstract method that should be implemented by subclasses to
        process the provided data.

        Parameters
        ----------
        data
            The data to be processed. The exact type and format will depend on
            the implementation.
        *args
            Variable length argument list for additional processing parameters.
        **kwargs
            Arbitrary keyword arguments for additional processing parameters.

        Returns
        -------
        The processed data, the format of which will depend on the implementation
        in the subclass.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "The process_data method must be implemented by subclasses."
        )
