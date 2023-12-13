import pandas as pd
from pandas import DataFrame

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
    >>> processed_df = sber_processor.process_data(df,)
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

    def create_matrix_(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a matrix from the given dataframe by converting 'cart' column values
        into dummy variables and grouping by 'user_id' and 'order_completed_at'.

        This method takes a dataframe, applies one-hot encoding to the 'cart' column
        creating dummy variables, and then groups the data by 'user_id' and
        'order_completed_at'. It returns a dataframe where each row represents a
        unique combination of 'user_id' and 'order_completed_at', with dummy
        variables indicating the presence of each cart item.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe with at least the columns 'user_id', 'order_completed_at',
            and 'cart'.

        Returns
        -------
        pd.DataFrame
            A transformed dataframe with one-hot encoded 'cart' items and grouped
            by 'user_id' and 'order_completed_at'.
        """
        df = pd.get_dummies(
            df, columns=["cart"], prefix="", prefix_sep="", dtype="bool"
        )
        df = df.groupby(["user_id", "order_completed_at"]).any().reset_index()
        return df

    def add_purchase_counter_(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a purchase counter to the dataframe, assigning a sequential order number
        to each user's purchases.

        This method adds an 'order_number' column to the dataframe, where each value
        represents the sequential number of a purchase for each user. It then removes
        the 'order_completed_at' column from the dataframe. The order number starts
        from 0 for each user's first purchase and increments by 1 for each subsequent
        purchase.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe with at least the columns 'user_id' and
            'order_completed_at'.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe with an added 'order_number' column, and without
            the 'order_completed_at' column.
        """
        df["order_number"] = df.groupby(["user_id"]).cumcount()
        df = df.drop("order_completed_at", axis=1)
        return df

    def split_dataset_(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the given dataframe into training and validation datasets.

        This method identifies the last order for each user and splits the dataframe into
        two: a training set and a validation set. The training set contains all orders
        except the last one for each user, while the validation set contains only the
        last order for each user. This is determined by the 'order_number' column.
        The training dataset groups data by 'user_id' and sums up the values of other
        columns.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe with at least the columns 'user_id' and 'order_number'.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple of two dataframes:
            - train: The training dataset, containing all but the last order for each user.
            - valid: The validation dataset, containing only the last order for each user.
        """
        last_order = (
            df.groupby(["user_id"])["order_number"].transform("max")
            == df["order_number"]
        )
        train = df[~last_order].groupby("user_id").sum().reset_index()
        valid = df[last_order].reset_index(drop=True)
        return train, valid

    def melt_dataframes_(
        self, train: pd.DataFrame, valid: pd.DataFrame
    ) -> tuple[DataFrame, DataFrame]:
        """
        Transform the training and validation dataframes into a long format.

        This method applies the pandas 'melt' function to both the training and validation
        dataframes, transforming them from a wide format to a long format. For the training
        dataframe, this involves melting all columns except 'user_id' into two new columns:
        'category' and 'ordered', where 'category' is the name of the original column and
        'ordered' is its value. For the validation dataframe, a similar process is followed,
        but the value column is named 'target'.

        Parameters
        ----------
        train : pd.DataFrame
            The training dataframe, with 'user_id' as one of the columns.
        valid : pd.DataFrame
            The validation dataframe, with 'user_id' as one of the columns.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple of two dataframes:
            - train_melt: The melted training dataframe in long format.
            - valid_melt: The melted validation dataframe in long format.
        """
        train_melt = pd.melt(
            train, id_vars=["user_id"], var_name="category", value_name="ordered"
        )
        valid_melt = pd.melt(
            valid, id_vars=["user_id"], var_name="category", value_name="target"
        )
        return train_melt, valid_melt

    def calculate_features_(
        self, train_melt: pd.DataFrame, valid_melt: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate additional features for the training dataset based on the validation dataset.

        This method computes additional features for the training dataset by using information
        from both the training and validation datasets. It copies the training dataset and
        then retrieves the maximum order number for each user from the validation dataset.
        It then calculates the total number of orders for each user and a rating feature,
        which is the ratio of 'ordered' to 'orders_total'. Finally, it creates a unique 'id'
        for each row, composed of the 'user_id' and 'category' separated by a semicolon.

        Parameters
        ----------
        train_melt : pd.DataFrame
            The melted training dataframe in long format, with 'user_id', 'category',
            and 'ordered' columns.
        valid_melt : pd.DataFrame
            The melted validation dataframe in long format, with at least 'user_id'
            and 'order_number' columns.

        Returns
        -------
        pd.DataFrame
            The enhanced training dataframe with additional features: 'orders_total',
            'rating', and 'id'.
        """
        train_df = train_melt.copy()
        order_number = (
            valid_melt[["user_id", "order_number"]].set_index("user_id").squeeze()
        )
        train_df["orders_total"] = train_df["user_id"].map(order_number)
        train_df["rating"] = train_df["ordered"] / train_df["orders_total"]
        train_df["id"] = train_df["user_id"].astype(str) + ";" + train_df["category"]
        return train_df

    def merge_and_filter_(
        self,
        train_df: pd.DataFrame,
        valid_melt: pd.DataFrame,
        submission_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge the training dataset with validation data and filter based on a submission dataframe.

        This method augments the training dataset (train_df) with target values from the validation
        dataset (valid_melt). It then filters the rows in the training dataset based on the
        unique 'id' values present in the submission dataframe. The 'target' column from the
        validation dataset is converted to integer type before merging. The final dataset
        contains only the entries that have corresponding 'id' values in the submission
        dataframe.

        Parameters
        ----------
        train_df : pd.DataFrame
            The training dataframe, which includes a unique 'id' for each row.
        valid_melt : pd.DataFrame
            The validation dataframe, containing at least the 'target' column.
        submission_df : pd.DataFrame
            A dataframe used for filtering, containing unique 'id' values.

        Returns
        -------
        pd.DataFrame
            The filtered and merged training dataframe, now containing 'target' values
            and rows corresponding to the 'id's in the submission dataframe.
        """
        train_df["target"] = valid_melt["target"].astype(int)
        train_df = train_df[train_df.id.isin(submission_df.id.unique())].reset_index(
            drop=True
        )
        return train_df

    def calculate_total_ordered_(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the total quantity ordered for each category in the training dataset.

        This method computes a new column 'total_ordered' in the training dataframe. It
        represents the total quantity of orders for each category across all users. This
        is achieved by summing up the 'ordered' column for each 'category' and mapping
        these sums back to the corresponding categories in the training dataset.

        Parameters
        ----------
        train_df : pd.DataFrame
            The training dataframe, containing at least 'category' and 'ordered' columns.

        Returns
        -------
        pd.DataFrame
            The modified training dataframe, now including a 'total_ordered' column
            indicating the total quantity ordered for each category.
        """
        total_ordered = train_df.groupby("category")["ordered"].sum()
        train_df["total_ordered"] = train_df["category"].map(total_ordered)
        return train_df

    def process_data(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Process raw data through various transformation steps
        to prepare it for analysis or modeling.

        This method orchestrates a series of data transformation steps on the input raw dataframe.
        It starts by creating dummy variables and grouping the data, then adds a purchase counter,
        splits the dataset into training and validation sets, melts these sets into a long format,
        calculates additional features, merges and filters the data based on a submission dataframe,
        and finally calculates the total quantity ordered for each category. The method is flexible
        to accept additional arguments and keyword arguments, which can be used in future
        enhancements or for passing additional parameters to the underlying methods.

        Parameters
        ----------
        data : pd.DataFrame
            The raw input dataframe to be processed.
        sub : pd.DataFrame
            The submission dataframe used for filtering in the merge_and_filter step.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        pd.DataFrame
            The processed dataframe, ready for further analysis or modeling, containing
            additional features and filtered based on the provided submission dataframe.
        """
        sub = kwargs.get("sub")
        grouped_dummies = self.create_matrix_(data)
        grouped_dummies_with_counter = self.add_purchase_counter_(grouped_dummies)
        train, valid = self.split_dataset_(grouped_dummies_with_counter)
        train_melt, valid_melt = self.melt_dataframes_(train, valid)
        train_df = self.calculate_features_(train_melt, valid)
        train_df = self.merge_and_filter_(train_df, valid_melt, sub)
        train_df = self.calculate_total_ordered_(train_df)
        return train_df
