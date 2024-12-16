
import pandas as pd
import pytest
from src.data_processor import GermanHousingDataProcessor

def test_preprocessing():
    # Create sample data
    data = pd.DataFrame({
        "obj_livingSpace": [50, 70, np.nan],
        "obj_noRooms": [2, 3, np.nan],
        "obj_yearConstructed": [2000, 1990, np.nan],
        "obj_purchasePrice": [100000, 150000, 120000]
    })

    processor = GermanHousingDataProcessor(data)
    X, y = processor.comprehensive_preprocessing()

    # Check for no missing values in X
    assert X.isnull().sum().sum() == 0
    # Check the target
    assert not y.isnull().any()
