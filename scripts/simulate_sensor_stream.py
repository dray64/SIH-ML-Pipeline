import time
import requests
import pandas as pd
from src.config import RAW_DATA_PATH

API_ENDPOINT = "http://127.0.0.1:8000/predict"

def run_simulation():
    df = pd.read_csv(RAW_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    surge_indices = df[df["gage_height_ft"] >= 15.0].index
    if len(surge_indices) == 0:
        surge_idx = len(df) - 15
    else:
        surge_idx = surge_indices[0]

    test_slice = df.iloc[max(0, surge_idx - 10) : min(len(df), surge_idx + 3)]

    print(f"[INFO] Streaming {len(test_slice)} sequential telemetry packets to {API_ENDPOINT}...\n")

    for _, row in test_slice.iterrows():
        payload = {
            "timestamp": row["timestamp"].isoformat(),
            "gage_height_ft": float(row["gage_height_ft"]),
            "precip_in": float(row["precip_in"]),
            "soil_moisture_pct": float(row["soil_moisture_pct"]),
            "temp_f": float(row["temp_f"]),
            "rel_humidity_pct": float(row["rel_humidity_pct"]),
            "altimeter_in_hg": float(row["altimeter_in_hg"]),
        }

        try:
            res = requests.post(API_ENDPOINT, json=payload)
            data = res.json()
            print(f"[{payload['timestamp'][:19]}] Water Level: {data['current_water_level_ft']:.2f} ft | "
                  f"+3h Forecast: {data['forecast_3h_ft']:.2f} ft | Rise: {data['projected_rate_of_rise_ft_hr']:+.2f} ft/h | "
                  f"[{data['alert_tier']}]")
        except requests.exceptions.ConnectionError:
            print("[ERROR] API server is not running. Start it with: uvicorn api.app:app --port 8000")
            break

        time.sleep(1.0)

if __name__ == "__main__":
    run_simulation()