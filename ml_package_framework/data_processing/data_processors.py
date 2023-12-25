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

    def create_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert categorical 'cart' values in the DataFrame into one-hot encoded columns,
        then group the data by 'user_id' and 'order_completed_at' and apply an 'any' aggregation.
        This transformation creates a matrix where
        each row represents a unique user-order combination,
        and each column represents the presence or absence of a specific item in the cart.

        Parameters
        ----------
        df : pd.DataFrame
            A pandas DataFrame containing the columns 'cart', 'user_id', and 'order_completed_at'.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row corresponds to a unique user-order pair, and each column
            represents a distinct item, indicating its presence or absence in that order.

        Raises
        ------
        ValueError
            If the input is not a pandas DataFrame or if required columns are missing.

        Examples
        --------
        >>> data = {'user_id': [1, 1, 2],
                    'order_completed_at': ['2021-01-01', '2021-01-02', '2021-01-02'],
                    'cart': ['1', '5', '1']}
        >>> df = pd.DataFrame(data)
        >>> processor = DataProcessor()
        >>> result = processor.create_user_item_matrix(df)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        required_columns = {"cart", "user_id", "order_completed_at"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        df = pd.get_dummies(
            df, columns=["cart"], prefix="", prefix_sep="", dtype="bool"
        )
        df = df.groupby(["user_id", "order_completed_at"]).any().reset_index()
        return df

    def add_purchase_counter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a sequential order number for each user's purchases in the DataFrame.

        This method computes a cumulative count of orders for each user, represented by 'user_id',
        and adds it as a new column 'order_number'. It then drops the 'order_completed_at' column
        from the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            A pandas DataFrame containing at least the columns 'user_id' and 'order_completed_at'.

        Returns
        -------
        pd.DataFrame
            The DataFrame with an added 'order_number' column and
            without the 'order_completed_at' column.

        Raises
        ------
        ValueError
            If the input is not a pandas DataFrame or if required columns are missing.

        Examples
        --------
        >>> data = {'user_id': [1, 1, 2, 2],
                    'order_completed_at': ['2021-01-01', '2021-01-02', '2021-01-01', '2021-01-02']}
        >>> df = pd.DataFrame(data)
        >>> processor = DataProcessor()
        >>> result = processor.add_purchase_counter(df)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        required_columns = {"user_id", "order_completed_at"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        df["order_number"] = df.groupby(["user_id"]).cumcount()
        df = df.drop("order_completed_at", axis=1)
        return df

    def split_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into two separate DataFrames based on the last order for each user.
        This method identifies the latest order for each user and splits the DataFrame into two:
        one DataFrame contains all data excluding
        the last order for each user, and the other includes
        only the last order for each user.
        The first DataFrame aggregates the data at the user level.

        Parameters
        ----------
        df : pd.DataFrame
            A pandas DataFrame containing at least the columns 'user_id' and 'order_number'.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple of two DataFrames. The first DataFrame (df_ex_last) contains all data excluding
            the last order for each user, aggregated at the user level. The second DataFrame
            (df_incl_last) includes only the last order for each user.

        Raises
        ------
        ValueError
            If the input is not a pandas DataFrame or if required columns are missing.

        Examples
        --------
        >>> data = {'user_id': [1, 1, 2, 2],
                    'order_number': [0, 1, 0, 1]}
        >>> df = pd.DataFrame(data)
        >>> processor = DataProcessor()
        >>> df_ex_last, df_incl_last = processor.split_dataset(df)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        required_columns = {"user_id", "order_number"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        last_order = (
            df.groupby(["user_id"])["order_number"].transform("max")
            == df["order_number"]
        )
        df_ex_last = df[~last_order].groupby("user_id").sum().reset_index()
        df_incl_last = df[last_order].reset_index(drop=True)

        return df_ex_last, df_incl_last

    def long_format_transform(
        self, df_ex_last: DataFrame, df_incl_last: DataFrame
    ) -> tuple[DataFrame, DataFrame]:
        """
        Transform two DataFrames from wide format to long format.

        This method takes two DataFrames, `df_ex_last` and `df_incl_last`, and transforms them from
        a wide format to a long format using pandas' melt function. In the resulting long format,
        'df_ex_last' will have 'ordered' as a value column, and 'df_incl_last' will have 'target' as
        a value column. Both DataFrames will have 'user_id' as an identifier and a 'category' column
        representing the original column names.

        Parameters
        ----------
        df_ex_last : DataFrame
            A pandas DataFrame in wide format to be transformed.
        df_incl_last : DataFrame
            Another pandas DataFrame in wide format to be transformed.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            A tuple containing the two transformed DataFrames in long format.

        Raises
        ------
        ValueError
            If either of the inputs is not a pandas DataFrame.

        Examples
        --------
        >>> df_ex_last = pd.DataFrame({
                'user_id': [1, 2],
                'apple': [1, 0],
                'banana': [0, 1]})
        >>> df_incl_last = pd.DataFrame({
                'user_id': [1, 2],
                'apple': [0, 1],
                'banana': [1, 0]})
        >>> processor = DataProcessor()
        >>> df_ex_last_long, df_incl_last_long = processor.long_format_transform(df_ex_last, df_incl_last)
        """
        if not isinstance(df_ex_last, pd.DataFrame) or not isinstance(
            df_incl_last, pd.DataFrame
        ):
            raise ValueError("Both inputs must be pandas DataFrames.")

        df_ex_last_long = pd.melt(
            df_ex_last, id_vars=["user_id"], var_name="category", value_name="ordered"
        )
        df_incl_last_long = pd.melt(
            df_incl_last, id_vars=["user_id"], var_name="category", value_name="target"
        )

        return df_ex_last_long, df_incl_last_long

    def calculate_features(
        self, df_ex_last_flatten: pd.DataFrame, df_incl_last: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate additional feature columns for a DataFrame.

        This method computes the total number of orders and a rating (ratio of ordered items to total orders)
        for each user-category pair in the dataset. It also constructs a unique ID combining user_id and category.

        Parameters
        ----------
        df_ex_last_flatten : pd.DataFrame
            A pandas DataFrame representing historical order data in long format.
        df_incl_last : pd.DataFrame
            A pandas DataFrame representing the most recent order data.

        Returns
        -------
        pd.DataFrame
            The DataFrame with additional calculated features: 'orders_total', 'rating', and 'id'.

        Examples
        --------
        >>> df_ex_last_flatten = pd.DataFrame({
                'user_id': [1, 2],
                'category': ['apple', 'banana'],
                'ordered': [1, 2]})
        >>> df_incl_last = pd.DataFrame({
                'user_id': [1, 2],
                'order_number': [3, 4]})
        >>> processor = DataProcessor()
        >>> features_df = processor.calculate_features(df_ex_last_flatten, df_incl_last)
        """
        order_number = (
            df_incl_last[["user_id", "order_number"]].set_index("user_id").squeeze()
        )
        train_df = df_ex_last_flatten.copy()
        train_df["orders_total"] = train_df["user_id"].map(order_number)
        train_df["rating"] = train_df["ordered"] / train_df["orders_total"]
        train_df["id"] = train_df["user_id"].astype(str) + ";" + train_df["category"]

        return train_df

    def merge_and_filter(
        self,
        train_df: pd.DataFrame,
        df_incl_last: pd.DataFrame,
        submission_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge training data with most recent orders and filter based on submission data.

        This method adds a 'target' column to the training dataset from `df_incl_last` and filters
        the training dataset to include only rows with IDs present in the submission dataset.

        Parameters
        ----------
        train_df : pd.DataFrame
            The training dataset with user, category, and other features.
        df_incl_last : pd.DataFrame
            The DataFrame containing the most recent order data.
        submission_df : pd.DataFrame
            A DataFrame with IDs used to filter the training dataset.

        Returns
        -------
        pd.DataFrame
            The filtered training dataset, now including a 'target' column.

        Examples
        --------
        >>> train_df = pd.DataFrame({
                'id': ['1;apple', '2;banana'],
                'user_id': [1, 2],
                'category': ['apple', 'banana'],
                'ordered': [1, 2]})
        >>> df_incl_last = pd.DataFrame({
                'user_id': [1, 2],
                'target': [0, 1]})
        >>> submission_df = pd.DataFrame({'id': ['1;apple']})
        >>> processor = DataProcessor()
        >>> filtered_df = processor.merge_and_filter(train_df, df_incl_last, submission_df)
        """
        train_df["target"] = df_incl_last["target"].astype(int)
        train_df = train_df[train_df.id.isin(submission_df.id.unique())].reset_index(
            drop=True
        )
        return train_df

    def calculate_total_ordered(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add the total ordered quantity for each category in the training dataset.

        This method computes the total quantity of items ordered in each category across all users
        and adds this total as a new column 'total_ordered' in the training dataset.

        Parameters
        ----------
        train_df : pd.DataFrame
            The training dataset with user, category, and ordered quantity information.

        Returns
        -------
        pd.DataFrame
            The training dataset enhanced with the 'total_ordered' column representing the total
            quantity ordered for each category.

        Examples
        --------
        >>> train_df = pd.DataFrame({
                'category': ['apple', 'banana', 'apple'],
                'ordered': [1, 2, 3]})
        >>> processor = DataProcessor()
        >>> enhanced_df = processor.calculate_total_ordered(train_df)
        """
        total_ordered = train_df.groupby("category")["ordered"].sum()
        train_df["total_ordered"] = train_df["category"].map(total_ordered)
        return train_df

    def process_data(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Process the input DataFrame through a series of transformation steps.

        This method applies a pipeline of data transformations, including creating a user-item matrix,
        adding a purchase counter, splitting the dataset, transforming format, calculating features,
        merging with submission data, and calculating total ordered quantities.

        Parameters
        ----------
        data : pd.DataFrame
            The initial dataset to be processed.
        **kwargs
            Additional keyword arguments, where 'sub' (submission DataFrame) can be passed.

        Returns
        -------
        pd.DataFrame
            The processed dataset, ready for further analysis or model training.

        Examples
        --------
        >>> raw_data = pd.DataFrame({...})  # Replace with actual data structure
        >>> submission_data = pd.DataFrame({...})  # Replace with actual data structure
        >>> processor = DataProcessor()
        >>> processed_data = processor.process_data(raw_data, sub=submission_data)
        """
        sub = kwargs.get("sub")
        intersection_matrix = self.create_user_item_matrix(data)
        intersection_matrix_with_counter = self.add_purchase_counter(
            intersection_matrix
        )
        df_ex_last, df_incl_last = self.split_dataset(intersection_matrix_with_counter)
        df_ex_last_flatten, df_incl_last_flatten = self.long_format_transform(
            df_ex_last, df_incl_last
        )
        train_df = self.calculate_features(df_ex_last_flatten, df_incl_last)
        train_df = self.merge_and_filter(train_df, df_incl_last_flatten, sub)
        train_df = self.calculate_total_ordered(train_df)
        return train_df
