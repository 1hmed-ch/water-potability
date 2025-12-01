from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os

from app.models.model import WaterPotabilityModel
from app.schemas.prediction import WaterFeatures, PredictionResponse

# Create FastAPI application
app = FastAPI(
    title="Water Potability Prediction API",
    description="API for predicting water potability using AI algorithms",
    version="1.0.0"
)

# Setup CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize model
print("🔄 Loading Water Potability Model...")
model = WaterPotabilityModel()
print("✅ Model initialization complete!")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Home page with API information"""
    return """
    <html>
        <head>
            <title>Water Potability API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌊 Water Potability Prediction API</h1>
                <p>Welcome to the Water Potability Prediction API!</p>
                
                <div class="endpoint">
                    <h3>📊 Available Endpoints:</h3>
                    <ul>
                        <li><strong>GET /</strong> - This page</li>
                        <li><strong>POST /predict</strong> - Predict water potability</li>
                        <li><strong>GET /health</strong> - Check server status</li>
                        <li><strong>GET /features</strong> - Show feature descriptions</li>
                    </ul>
                </div>
                
                <p>📚 Use <strong>POST /predict</strong> with water data to get predictions</p>
                <p>🔗 <strong>Interactive documentation:</strong> <a href="/docs">/docs</a></p>
                <p>🔗 <strong>Alternative documentation:</strong> <a href="/redoc">/redoc</a></p>
            </div>
        </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResponse)
async def predict_potability(water_data: WaterFeatures):
    """
    Predict water potability based on chemical characteristics
    
    - **ph**: pH value (Safe range: 6.5-8.5)
    - **Hardness**: Water hardness (mg/L)
    - **Solids**: Total Dissolved Solids (ppm)
    - **Chloramines**: Chloramine level (ppm)
    - **Sulfate**: Sulfate content (mg/L)
    - **Conductivity**: Electrical conductivity (μS/cm)
    - **Organic_carbon**: Total Organic Carbon (ppm)
    - **Trihalomethanes**: THM level (ppm)
    - **Turbidity**: Water turbidity (NTU)
    """
    try:
        print("🎯 Received prediction request")
        result = model.predict(water_data.dict())
        return result
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    """Check server and model status"""
    return {
        "status": "healthy",
        "message": "Server is running normally",
        "model_loaded": model.model is not None,
        "service": "Water Potability Prediction API"
    }

@app.get("/features")
async def get_features_info():
    """Explain the meaning of each water feature"""
    features_info = {
        "ph": "pH value (Safe range: 6.5 - 8.5)",
        "Hardness": "Hardness (caused by calcium and magnesium salts, measured in mg/L)",
        "Solids": "Total Dissolved Solids - TDS (total mineral content, measured in ppm)",
        "Chloramines": "Chloramines (disinfectant, Safe level: up to 4 ppm)",
        "Sulfate": "Sulfate (natural substance from minerals and soil, measured in mg/L)",
        "Conductivity": "Electrical conductivity (related to dissolved solids, measured in μS/cm)",
        "Organic_carbon": "Organic Carbon - TOC (Lower values are better, measured in ppm)",
        "Trihalomethanes": "Trihalomethanes (byproduct of chlorination, Safe level: up to 80 ppm)",
        "Turbidity": "Turbidity (indicates suspended particles, measured in NTU)"
    }
    return features_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)