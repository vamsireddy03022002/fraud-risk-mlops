import pandas as pd

from fraud.features import FeatureConfig, build_features


def test_feature_pipeline_smoke():
    df = pd.DataFrame(
        {
            "trans_date_trans_time": ["2019-01-01 01:00:00"],
            "cc_num": [1],
            "merchant": ["m1"],
            "category": ["shopping"],
            "amt": [100.0],
            "gender": ["M"],
            "zip": [12345],
            "lat": [40.0],
            "long": [-70.0],
            "city_pop": [100000],
            "unix_time": [1546300800],
            "merch_lat": [41.0],
            "merch_long": [-71.0],
        }
    )
    X = build_features(df, config=FeatureConfig(category_top_k=10))
    assert X.shape[0] == 1
