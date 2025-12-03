""" Random Forest model with hyperparameter tuning and imputation
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer


def train_and_save_model():
    print("🚀 Starting Random Forest model training...")

    # 1. Load Data
    if not os.path.exists('data/water_potability.csv'):
        print("❌ Error: 'data/water_potability.csv' not found.")
        return

    df = pd.read_csv('data/water_potability.csv')

    # 2. Smart Imputation
    # Random Forest handles unscaled data well, but we still need to fill missing values
    print("⚙️  Handling missing values...")
    imputer = KNNImputer(n_neighbors=5)

    # Separate X and y
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    # Impute before splitting to keep it simple for this step,
    # or strictly: split -> fit imputer on train -> transform test.
    # We will do strict split to ensure 100% validity.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit Imputer on Train, Transform Test
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    # 3. Hyperparameter Tuning for Random Forest
    print("🔍 Tuning hyperparameters (Random Forest)...")

    # Random Forest is robust and often beats SVM on tabular data
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Use GridSearchCV to find best settings
    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"✅ Best Params: {grid.best_params_}")

    # 4. Evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Real Model Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Save
    os.makedirs('app/models', exist_ok=True)
    joblib.dump(best_model, 'app/models/model.pkl')
    # Note: Random Forest doesn't strictly need a scaler,
    # but if you want to keep the API consistent, we can save the imputer or a dummy scaler.
    # For this app, we usually just save the model.
    print("💾 Model saved to 'app/models/model.pkl'")


if __name__ == "__main__":
    train_and_save_model()
"""

""" SVM model with hyperparameter tuning and scaling
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_and_save_model():
    print("🚀 Starting optimized model training...")

    # 1. Load Data
    if not os.path.exists('data/water_potability.csv'):
        print("❌ Error: 'data/water_potability.csv' not found.")
        return

    df = pd.read_csv('data/water_potability.csv')
    print(f"✅ Data loaded: {df.shape[0]} samples")

    # 2. Smart Imputation (Class-based filling)
    # Instead of a generic mean, we fill missing values based on the Potability class.
    # This preserves the data distribution better than a global mean.
    print("⚙️  Handling missing values...")
    for col in ['ph', 'Sulfate', 'Trihalomethanes']:
        df[col] = df[col].fillna(df.groupby('Potability')[col].transform('mean'))

    # Drop any remaining rows with missing values just in case
    df.dropna(inplace=True)

    # 3. Split Data
    # Stratify ensures both train and test sets have the same proportion of Potable/Not Potable water
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scaling (Crucial for SVM)
    # We fit the scaler ONLY on the training data to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Hyperparameter Tuning
    # We use GridSearchCV to find the best parameters for the SVM
    print("🔍 Tuning hyperparameters (this may take a moment)...")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf']
    }

    # class_weight='balanced' helps handling the imbalance in your dataset
    grid = GridSearchCV(
        SVC(probability=True, class_weight='balanced', random_state=42),
        param_grid,
        refit=True,
        verbose=1,
        cv=3,
        n_jobs=-1  # Use all available CPU cores
    )

    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    print(f"✅ Best Params: {grid.best_params_}")

    # 6. Evaluation
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Model Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Artifacts
    os.makedirs('app/models', exist_ok=True)
    joblib.dump(best_model, 'app/models/model.pkl')
    joblib.dump(scaler, 'app/models/scaler.pkl')
    print("💾 Model and Scaler saved to 'app/models/'")


if __name__ == "__main__":
    train_and_save_model()

"""

""" Random Forest model replaced with LightGBM and advanced feature engineering 
import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def train_and_save_model():
    print("🚀 Starting LightGBM + Feature Engineering pipeline...")

    # 1. Load Data
    if not os.path.exists('data/water_potability.csv'):
        print("❌ Error: 'data/water_potability.csv' not found.")
        return

    df = pd.read_csv('data/water_potability.csv')

    # 2. Advanced Preprocessing
    print("🧪 Performing Chemical Feature Engineering...")

    # Fill missing values first (KNN is best for this)
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(df.drop('Potability', axis=1)),
                             columns=df.columns[:-1])

    # --- FEATURE ENGINEERING START ---
    # Create interactions that mimic chemical properties
    # 1. pH * Hardness: Affects mineral solubility
    X_imputed['ph_hardness'] = X_imputed['ph'] * X_imputed['Hardness']
    # 2. Sulfate / Hardness: Ratio of permanent hardness
    X_imputed['sulfate_hardness'] = X_imputed['Sulfate'] / X_imputed['Hardness']
    # 3. Chloramines * Turbidity: Disinfectant efficiency vs cloudiness
    X_imputed['chlor_turbidity'] = X_imputed['Chloramines'] * X_imputed['Turbidity']
    # --- FEATURE ENGINEERING END ---

    y = df['Potability']
    X = X_imputed

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scaling (Good for convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. LightGBM Setup
    print("⚡ Tuning LightGBM (this is usually faster than XGBoost)...")

    lgbm = lgb.LGBMClassifier(random_state=42, verbose=-1)

    # Search a wider parameter space
    param_dist = {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 50],  # Key parameter for LightGBM
        'max_depth': [-1, 10, 20],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],  # L1 Regularization
        'reg_lambda': [0, 0.1, 0.5],  # L2 Regularization
        'class_weight': ['balanced', None]  # Test both balanced and unbalanced
    }

    # RandomizedSearch is faster and often finds better params than Grid
    random_search = RandomizedSearchCV(
        lgbm,
        param_distributions=param_dist,
        n_iter=50,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train_scaled, y_train)

    best_model = random_search.best_estimator_
    print(f"✅ Best Params: {random_search.best_params_}")

    # 6. Evaluation
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 LightGBM Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save
    os.makedirs('app/models', exist_ok=True)
    joblib.dump(best_model, 'app/models/model.pkl')
    # Save scaler too as we need it for inference now
    joblib.dump(scaler, 'app/models/scaler.pkl')
    print("💾 Model and Scaler saved to 'app/models/'")


if __name__ == "__main__":
    train_and_save_model()
"""

# Random Forest + LightGBM Stacking Ensemble with Feature Engineering
import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_and_save_model():
    print("Starting Stacking Ensemble (RF + LightGBM)...")

    # 1. Load Data
    if not os.path.exists('data/water_potability.csv'):
        print("Error: 'data/water_potability.csv' not found.")
        return

    df = pd.read_csv('data/water_potability.csv')

    # 2. Preprocessing & Feature Engineering
    print("Processing data...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(df.drop('Potability', axis=1)),
                             columns=df.columns[:-1])

    # Re-apply the features that worked well
    X_imputed['ph_hardness'] = X_imputed['ph'] * X_imputed['Hardness']
    X_imputed['sulfate_hardness'] = X_imputed['Sulfate'] / X_imputed['Hardness']
    X_imputed['chlor_turbidity'] = X_imputed['Chloramines'] * X_imputed['Turbidity']

    y = df['Potability']
    X = X_imputed

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Define Base Models
    # We use pipelines to ensure scaling happens correctly inside the stack

    # Base Model 1: Random Forest (The reliable one)
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    # Base Model 2: LightGBM (The gradient booster)
    # We use the best params you found in your last run
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=50,
        max_depth=20,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=1.0,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1
    )

    # LightGBM needs scaling, RF doesn't strictly, but good to keep consistent
    lgbm_pipe = make_pipeline(StandardScaler(), lgbm)
    rf_pipe = make_pipeline(StandardScaler(), rf)

    estimators = [
        ('rf', rf_pipe),
        ('lgbm', lgbm_pipe)
    ]

    # 5. Define Stacking Classifier
    # The 'final_estimator' learns how to combine the base models
    print("Building the Stack...")
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,  # Cross-validation for the meta-learner
        n_jobs=-1
    )

    # 6. Train
    print("⚡ Training... (This combines both models)")
    stacking_clf.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = stacking_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nStacking Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Save
    os.makedirs('app/models', exist_ok=True)
    joblib.dump(stacking_clf, 'app/models/model.pkl')
    print("Stacked Model saved to 'app/models/model.pkl'")


if __name__ == "__main__":
    train_and_save_model()
