import numpy as np
import pandas as pd

def build_features_from_dataframe(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Computes all 35 hydrometeorological features identical across training & inference.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Hydraulic velocity and acceleration
    df["gage_height_change_1h"] = df["gage_height_ft"].diff().fillna(0.0)
    df["stage_velocity"] = df["gage_height_change_1h"]
    df["stage_accel"] = df["gage_height_change_1h"].diff().fillna(0.0)

    # 2. Cumulative precipitation windows
    df["precip_cum_3h"] = df["precip_in"].rolling(3, min_periods=1).sum()
    df["precip_cum_6h"] = df["precip_in"].rolling(6, min_periods=1).sum()
    df["precip_cum_24h"] = df["precip_in"].rolling(24, min_periods=1).sum()

    # 3. Temporal lag structures
    lags = [1, 2, 3, 6, 12]
    for l in lags:
        df[f"stage_lag_{l}h"] = df["gage_height_ft"].shift(l).bfill()
        df[f"precip_lag_{l}h"] = df["precip_in"].shift(l).bfill()
        df[f"soil_lag_{l}h"] = df["soil_moisture_pct"].shift(l).bfill()

    # 4. Infiltration and catchment momentum
    df["precip_ewma_6h"] = df["precip_in"].ewm(span=6).mean()
    df["precip_ewma_24h"] = df["precip_in"].ewm(span=24).mean()
    df["stage_ewma_6h"] = df["gage_height_ft"].ewm(span=6).mean()
    df["stage_diff_3h"] = df["gage_height_ft"] - df["stage_lag_3h"]
    df["stage_diff_6h"] = df["gage_height_ft"] - df["stage_lag_6h"]
    df["precip_max_6h"] = df["precip_in"].rolling(6, min_periods=1).max()
    df["soil_delta_1h"] = df["soil_moisture_pct"] - df["soil_lag_1h"]
    df["runoff_potential_3h"] = df["precip_cum_3h"] * (df["soil_moisture_pct"] / 100.0)

    if is_training:
        df["target_1h"] = df["gage_height_ft"].shift(-1)
        df["target_2h"] = df["gage_height_ft"].shift(-2)
        df["target_3h"] = df["gage_height_ft"].shift(-3)
        return df.dropna().reset_index(drop=True)

    return df