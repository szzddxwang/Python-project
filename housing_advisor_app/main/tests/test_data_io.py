import pytest
import pandas as pd
from data_io import load_raw_sales

def test_load_raw_sales_returns_dataframe():
    df = load_raw_sales()
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) > 0
    assert len(df) > 0
