import pandas as pd

from fraud.features import FeatureConfig, build_features, split_xy


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trans_date_trans_time": ["2019-01-01 01:00:00", "2019-01-02 13:00:00"],
            "cc_num": [1, 2],
            "merchant": ["m1", "m2"],
            "category": ["shopping", "food"],
            "amt": [100.0, 20.0],
            "gender": ["M", "F"],
            "zip": [12345, 12345],
            "lat": [40.0, 40.0],
            "long": [-70.0, -70.0],
            "city_pop": [100000, 100000],
            "unix_time": [1546300800, 1546387200],
            "merch_lat": [41.0, 41.0],
            "merch_long": [-71.0, -71.0],
            "is_fraud": [0, 1],
        }
    )


def test_split_xy_removes_label():
    df = _sample_df()
    X_raw, y = split_xy(df)
    assert "is_fraud" not in X_raw.columns
    assert y.shape[0] == df.shape[0]


def test_build_features_no_label_leakage():
    df = _sample_df()
    X_raw, _ = split_xy(df)
    X_feat = build_features(X_raw, config=FeatureConfig(category_top_k=10))
    assert "is_fraud" not in X_feat.columns


def test_build_features_deterministic_columns():
    df = _sample_df()
    X_raw, _ = split_xy(df)

    X1 = build_features(X_raw, config=FeatureConfig(category_top_k=10))
    X2 = build_features(X_raw, config=FeatureConfig(category_top_k=10))

    assert list(X1.columns) == list(X2.columns)
    assert X1.shape == X2.shape
