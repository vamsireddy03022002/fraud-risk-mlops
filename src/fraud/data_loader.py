import pandas as pd
import yaml


def load_schema(schema_path: str) -> dict:
    with open(schema_path, "r") as file:
        return yaml.safe_load(file)


def load_data(csv_path: str, schema_path: str) -> pd.DataFrame:
    schema = load_schema(schema_path)
    expected_columns = schema["columns"]

    df = pd.read_csv(csv_path)

    # Drop accidental index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 1️⃣ Check for missing columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # 2️⃣ Enforce column order (optional but good discipline)
    df = df[expected_columns]

    # 3️⃣ Enforce dtypes
    for col, expected_dtype in schema["dtypes"].items():
        try:
            df[col] = df[col].astype(expected_dtype)
        except Exception as exc:
            raise TypeError(
                f"Column '{col}' cannot be cast to {expected_dtype}"
            ) from exc

    # 4️⃣ Enforce non-null constraints
    for col in schema.get("non_nullable", []):
        if df[col].isnull().any():
            raise ValueError(f"Null values found in non-nullable column: {col}")

    return df
