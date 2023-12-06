import os
import tempfile

import pandas as pd

from ml_package_framework.data_processing import SberDataProcessor


def create_temp_csv(content: str) -> str:
    """
    Create a temporary CSV file with the given content.

    :param content: A string representing the content of the CSV file.
    :return: The path to the temporary CSV file.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv"
    ) as temp_file:
        with open(temp_file.name, "w", encoding="utf-8") as file:
            file.write(content)
            return temp_file.name


def test_load_data_with_temporary_file():
    """
    Test the load_data method of SberDataProcessor.

    This test checks if the SberDataProcessor correctly loads data
    from a CSV file into a pandas DataFrame.
    It involves creating a temporary CSV file with predefined content, loading this file using the
    SberDataProcessor, and then verifying the contents and structure of the resulting DataFrame.
    Finally, it cleans up by removing the temporary file.
    """
    # Setup
    processor = SberDataProcessor()
    csv_content = "col1,col2\nvalue1,value2\nvalue3,value4"
    temp_csv_path = create_temp_csv(csv_content)

    # Execute
    data = processor.load_data(temp_csv_path)

    # Verify
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2  # Two rows of data
    assert list(data.columns) == ["col1", "col2"]
    assert data.iloc[0]["col1"] == "value1"
    assert data.iloc[1]["col2"] == "value4"

    # Cleanup
    os.remove(temp_csv_path)
