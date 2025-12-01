from pydantic import BaseModel

class WaterFeatures(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    message: str
    confidence: str