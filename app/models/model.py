import joblib
import pandas as pd
import numpy as np
import os

class WaterPotabilityModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the saved model and scaler"""
        try:
            # Get the absolute path to the model files
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, 'model.pkl')
            scaler_path = os.path.join(current_dir, 'scaler.pkl')
            
            print(f"📂 Loading model from: {model_path}")
            print(f"📂 Loading scaler from: {scaler_path}")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure you've run train_model.py first!")
            raise e
    
    def predict(self, features: dict) -> dict:
        """Predict water potability"""
        try:
            # Convert data to DataFrame (maintain correct column order)
            feature_order = [
                'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
            ]
            
            # Create DataFrame with correct column order
            input_data = pd.DataFrame([[
                features['ph'],
                features['Hardness'],
                features['Solids'],
                features['Chloramines'],
                features['Sulfate'],
                features['Conductivity'],
                features['Organic_carbon'],
                features['Trihalomethanes'],
                features['Turbidity']
            ]], columns=feature_order)
            
            print("🔍 Input data received:")
            for key, value in features.items():
                print(f"   {key}: {value}")
            
            # Apply scaling
            scaled_data = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(scaled_data)[0]
            probability = self.model.predict_proba(scaled_data)[0]
            
            # Create explanatory message
            confidence = max(probability)
            if prediction == 1:
                message = "Water is POTABLE (safe to drink)"
                confidence_level = "High" if confidence > 0.8 else "Medium"
            else:
                message = "Water is NOT POTABLE (unsafe to drink)"
                confidence_level = "High" if confidence > 0.8 else "Medium"
            
            result = {
                "prediction": int(prediction),
                "probability": float(confidence),
                "message": message,
                "confidence": confidence_level
            }
            
            print(f"🎯 Prediction result: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise e