import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

def train_and_save_model():
    print("🚀 Starting model training...")
    
    # 1. Load data
    df = pd.read_csv('data/water_potability.csv')
    print("✅ Data loaded successfully")
    print(f"📊 Dataset shape: {df.shape}")
    
    # 2. Clean data
    df.fillna(df.mean(), inplace=True)
    print("✅ Data cleaned (missing values filled)")
    
    # 3. Separate features and target
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    print("✅ Features and target separated")
    print(f"📈 Class distribution: {y.value_counts().to_dict()}")
    
    # 4. Scale data (without balancing first)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    print("✅ Data scaled")
    
    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✅ Data split into train/test sets")
    
    # 6. Train model with class_weight to handle imbalance
    model = SVC(
        kernel='rbf', 
        C=10, 
        gamma='scale',
        random_state=42, 
        probability=True,
        class_weight='balanced'  # This handles imbalance
    )
    model.fit(X_train, y_train)
    print("✅ Model trained successfully")
    
    # 7. Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Create models directory if it doesn't exist
    os.makedirs('app/models', exist_ok=True)
    
    # 9. Save model and scaler
    joblib.dump(model, 'app/models/model.pkl')
    joblib.dump(scaler, 'app/models/scaler.pkl')
    
    print("💾 Model and scaler saved successfully!")
    print("📁 Files created:")
    print("   - app/models/model.pkl")
    print("   - app/models/scaler.pkl")
    
    # Show some predictions
    print("\n🔍 Sample predictions:")
    sample_predictions = model.predict_proba(X_test[:5])
    for i, probs in enumerate(sample_predictions):
        pred_class = model.predict([X_test[i]])[0]
        print(f"   Sample {i+1}: Class {pred_class} - Probabilities: {probs}")

if __name__ == "__main__":
    train_and_save_model()