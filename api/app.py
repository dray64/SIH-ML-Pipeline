from fastapi import FastAPI, HTTPException
from api.schemas import SensorReadingSchema, PredictionResponseSchema
from src.inference_engine import EarlyWarningPredictor

app = FastAPI(
    title="Flash Flood Early Warning REST API",
    version="1.0.0",
    description="Dual-stage gated machine learning inference engine for rapid flash-flood onset prediction"
)

predictor = None

@app.on_event("startup")
def init_engine():
    global predictor
    try:
        predictor = EarlyWarningPredictor()
        print("[INFO] Model engine loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load predictor: {e}")

@app.post("/predict", response_model=PredictionResponseSchema)
def predict_endpoint(payload: SensorReadingSchema):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictive model not initialized")
    
    result = predictor.process_reading(payload.dict())
    return {
        "status": "success",
        **result
    }

@app.get("/health")
def health_check():
    buffer_len = len(predictor.buffer) if predictor else 0
    return {"status": "healthy", "buffer_depth": buffer_len}

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Flash Flood Early Warning System</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; color: #333; }
                .card { background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 600px; }
                h1 { color: #0066cc; }
                a { display: inline-block; margin-top: 15px; padding: 10px 18px; background: #0066cc; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
                a:hover { background: #004c99; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🌊 Flash Flood Early Warning Engine</h1>
                <p>Status: <strong>Online & Operational</strong></p>
                <p>Telemetry ingest endpoint: <code>POST /predict</code></p>
                <a href="/docs">Open Interactive API Docs</a>
            </div>
        </body>
    </html>
    """