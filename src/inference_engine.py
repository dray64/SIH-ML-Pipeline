import json
import joblib
import pandas as pd
from pathlib import Path
from src.config import (
    MODEL_PATH,
    CACHE_BUFFER_PATH,
    MAX_BUFFER_HOURS,
    ACTION_STAGE_FT,
    MAJOR_FLOOD_STAGE_FT,
    ADVISORY_STAGE_FT,
    CRITICAL_RISE_RATE_FT_HR,
    WARNING_RISE_RATE_FT_HR,
)
from src.feature_pipeline import build_features_from_dataframe

class EarlyWarningPredictor:
    def __init__(self, model_file: Path = MODEL_PATH):
        if not model_file.exists():
            raise FileNotFoundError(f"Model bundle '{model_file}' not found. Please train first.")
        self.bundle = joblib.load(model_file)
        self.features = self.bundle["feature_names"]
        self.models = self.bundle["models"]
        self.buffer = self._load_cache()

    def _load_cache(self):
        if CACHE_BUFFER_PATH.exists():
            try:
                with open(CACHE_BUFFER_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _persist_cache(self):
        CACHE_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_BUFFER_PATH, "w") as f:
            json.dump(self.buffer, f)

    def process_reading(self, reading: dict) -> dict:
        self.buffer.append(reading)
        if len(self.buffer) > MAX_BUFFER_HOURS:
            self.buffer.pop(0)
        self._persist_cache()

        df_history = pd.DataFrame(self.buffer)
        df_feats = build_features_from_dataframe(df_history, is_training=False)
        input_vector = df_feats.iloc[-1:][self.features]

        forecasts = {}
        probabilities = {}
        for h in [1, 2, 3]:
            prob = float(self.models[h]["gate"].predict_proba(input_vector)[0, 1])
            p_base = float(self.models[h]["base"].predict(input_vector)[0])
            p_surge = float(self.models[h]["surge"].predict(input_vector)[0])
            forecasts[h] = round((1.0 - prob) * p_base + prob * p_surge, 2)
            probabilities[h] = round(prob * 100.0, 1)

        curr_stage = float(reading["gage_height_ft"])
        peak_pred = max(forecasts.values())
        rate_of_rise = round((forecasts[3] - curr_stage) / 3.0, 2)

        if peak_pred >= MAJOR_FLOOD_STAGE_FT or rate_of_rise >= CRITICAL_RISE_RATE_FT_HR:
            tier = "CRITICAL: MAJOR FLASH FLOOD"
            action = "Evacuate low-lying riverbanks immediately."
        elif peak_pred >= ACTION_STAGE_FT or rate_of_rise >= WARNING_RISE_RATE_FT_HR:
            tier = "WARNING: ACTION / FLOOD STAGE"
            action = "Close low-water crossings and trigger sirens."
        elif peak_pred >= ADVISORY_STAGE_FT:
            tier = "ADVISORY: ELEVATED FLOW"
            action = "Maintain continuous automated telemetry surveillance."
        else:
            tier = "NORMAL (BASEFLOW)"
            action = "Nominal conditions. No threat detected."

        return {
            "timestamp": reading["timestamp"],
            "current_water_level_ft": curr_stage,
            "forecast_1h_ft": forecasts[1],
            "forecast_2h_ft": forecasts[2],
            "forecast_3h_ft": forecasts[3],
            "projected_rate_of_rise_ft_hr": rate_of_rise,
            "surge_probability_3h": probabilities[3],
            "alert_tier": tier,
            "action_protocol": action,
            "lead_time_hours": 3.0
        }