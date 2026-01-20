import pandas as pd
import yaml


def load_schema(schema_path: str) -> dict:
    with open(schema_path, "r") as file:
        return yaml.safe_load(file)


def load_data(csv_path: str, schema_path: str) -> pd.DataFrame:
    schema = load_schema(schema_path)

    df = pd.read_csv(csv_path)

    # Drop accidental index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Check for missing required columns
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select only schema-defined columns (feature contract)
    df = df[list(schema.keys())]

    for column, rules in schema.items():
        expected_dtype = rules["dtype"]

        # Datetime parsing
        if expected_dtype == "datetime":
            df[column] = pd.to_datetime(
                df[column],
                errors="raise",
            )

        # Null check
        if not rules["nullable"] and df[column].isnull().any():
            raise ValueError(f"Null values found in non-nullable column: {column}")

        # Type validation
        if expected_dtype == "float":
            if not pd.api.types.is_float_dtype(df[column]):
                raise TypeError(f"Column {column} must be float")

        elif expected_dtype == "int":
            if not pd.api.types.is_integer_dtype(df[column]):
                raise TypeError(f"Column {column} must be int")

        elif expected_dtype == "string":
            if not pd.api.types.is_object_dtype(df[column]):
                raise TypeError(f"Column {column} must be string")

    return df
