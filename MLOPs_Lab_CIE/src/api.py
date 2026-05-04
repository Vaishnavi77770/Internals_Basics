from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

app = FastAPI()

# Input validation
class InputData(BaseModel):
    attack_surface_count: int = Field(..., ge=5, le=100)
    patch_age_days: int = Field(..., ge=1, le=365)
    is_external_facing: int = Field(..., ge=0, le=1)
    tech_stack_complexity: int = Field(..., ge=1, le=5)

# Health endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/forecast")
def forecast(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = df.mean(axis=1)[0]   # simple logic
    return {"prediction": prediction}