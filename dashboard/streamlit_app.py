import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Flash Flood Early Warning System", layout="wide")

st.title("🌊 Blanco Basin Flash Flood Early Warning System")
st.markdown("Real-time telemetry ingestion and multi-horizon surge forecasting.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simulate Field Sensor Telemetry")
    stage = st.slider("River Stage (ft)", 2.0, 35.0, 5.25)
    precip = st.slider("Past 1h Rain (inches)", 0.0, 5.0, 1.84)
    soil = st.slider("Soil Moisture Saturation (%)", 5.0, 98.0, 48.0)
    temp = st.number_input("Temperature (°F)", value=72.0)
    humidity = st.number_input("Relative Humidity (%)", value=89.0)
    altimeter = st.number_input("Barometer (in Hg)", value=29.85)

    submit = st.button("Transmit Reading to API")

if submit:
    payload = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "gage_height_ft": stage,
        "precip_in": precip,
        "soil_moisture_pct": soil,
        "temp_f": temp,
        "rel_humidity_pct": humidity,
        "altimeter_in_hg": altimeter,
    }

    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=payload)
        data = res.json()

        with col2:
            st.subheader("Predictive Early-Warning Output")
            tier = data["alert_tier"]
            if "CRITICAL" in tier:
                st.error(f"🚨 **{tier}**")
            elif "WARNING" in tier:
                st.warning(f"⚠️ **{tier}**")
            else:
                st.success(f"✅ **{tier}**")

            st.write(f"**Action Protocol:** {data['action_protocol']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("+1h Horizon", f"{data['forecast_1h_ft']:.2f} ft")
            m2.metric("+2h Horizon", f"{data['forecast_2h_ft']:.2f} ft")
            m3.metric("+3h Horizon", f"{data['forecast_3h_ft']:.2f} ft", f"{data['projected_rate_of_rise_ft_hr']:+.2f} ft/hr")

            chart_data = pd.DataFrame({
                "Lead Time": ["Current", "+1h", "+2h", "+3h"],
                "Stage (ft)": [data["current_water_level_ft"], data["forecast_1h_ft"], data["forecast_2h_ft"], data["forecast_3h_ft"]]
            })
            st.line_chart(chart_data.set_index("Lead Time"))
    except Exception as e:
        st.error(f"Could not reach API. Ensure `uvicorn api.app:app` is running. ({e})")