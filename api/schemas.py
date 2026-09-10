from pydantic import BaseModel, Field

class SensorReadingSchema(BaseModel):
    timestamp: str = Field(..., example="2026-09-09T13:00:00Z")
    gage_height_ft: float = Field(..., description="Current river stage height in feet", example=5.25)
    precip_in: float = Field(..., description="Rainfall depth in inches during past hour", example=1.84)
    soil_moisture_pct: float = Field(..., description="Calibrated volumetric water content percentage", example=47.9)
    temp_f: float = Field(..., example=72.0)
    rel_humidity_pct: float = Field(..., example=89.0)
    altimeter_in_hg: float = Field(..., example=29.85)

class PredictionResponseSchema(BaseModel):
    status: str
    current_water_level_ft: float
    forecast_1h_ft: float
    forecast_2h_ft: float
    forecast_3h_ft: float
    projected_rate_of_rise_ft_hr: float
    surge_probability_3h: float
    alert_tier: str
    action_protocol: str
    lead_time_hours: float