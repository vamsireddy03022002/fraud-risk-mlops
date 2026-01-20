import pytest

from fraud.data_loader import load_data

SCHEMA_PATH = "configs/schema.yaml"


def test_valid_data_loads():
    df = load_data("tests/data/good.csv", SCHEMA_PATH)
    assert not df.empty
    assert "is_fraud" in df.columns


def test_missing_column_raises():
    with pytest.raises(ValueError):
        load_data("tests/data/missing_col.csv", SCHEMA_PATH)


def test_wrong_dtype_raises():
    with pytest.raises(TypeError):
        load_data("tests/data/wrong_dtype.csv", SCHEMA_PATH)
