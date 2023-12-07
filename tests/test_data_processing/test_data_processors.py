import os
import tempfile

import pandas as pd
import pytest

from ml_package_framework.data_processing import SberDataProcessor


@pytest.fixture(name="sample_data")
def fixture_sample_data():
    """
    Create a sample pandas DataFrame for use as a pytest fixture.

    This fixture generates a DataFrame with predefined data. It's primarily intended
    for use in testing contexts where a consistent and simple DataFrame is needed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns ('col1' and 'col2') and two rows of sample data.
        The values in 'col1' are ['value1', 'value2'], and the values in 'col2' are
        ['value3', 'value4'].
    """
    data = {"col1": ["value1", "value2"], "col2": ["value3", "value4"]}

    return pd.DataFrame(data)


def test_load_data_with_temporary_file(sample_data):
    """
    Test the functionality of the SberDataProcessor class for loading data from a CSV file.

    This test function validates whether the `load_data` method of the SberDataProcessor
    class correctly reads data from a CSV file and whether the data read matches the
    provided sample data. The test involves creating a temporary CSV file with the
    sample data, using the SberDataProcessor to load this data, and then comparing the
    loaded data against the original sample data.

    Parameters
    ----------
    sample_data : pandas.DataFrame
        A pandas DataFrame provided as a fixture. It contains the sample data for testing,
        which is written to a CSV file and used for comparison with the loaded data.

    Raises
    ------
    AssertionError
        If the DataFrame loaded by the SberDataProcessor does not match the
        sample_data DataFrame. This indicates a failure in the data loading process.

    Notes
    -----
    The test passes if the `pd.testing.assert_frame_equal` function confirms that
    the loaded DataFrame is identical to the sample DataFrame. This ensures the
    integrity and correctness of the data loading process implemented in the
    SberDataProcessor class.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        dst_path = os.path.join(temp_dir, "data.csv")
        sample_data.to_csv(dst_path, index=False)

        processor = SberDataProcessor()
        loaded_data = processor.load_data(dst_path)

        pd.testing.assert_frame_equal(loaded_data, sample_data)
