import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_models(data_path='data/cleaned_housing.csv', preprocessor_path='models/preprocessor.joblib', models_dir='models'):
    print(f"Loading cleaned data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # We must drop rows where 'Price' is NaN since it's the target variable
    initial_len = len(df)
    df.dropna(subset=['Price'], inplace=True)
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing target 'Price'.")
        
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Load preprocessor
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    
    # Split dataset (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transform features
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost Regressor': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_model_name = ""
    
    print("\n--- Model Training & Evaluation ---")
    for name, model in models.items():
        # Train
        model.fit(X_train_processed, y_train)
        
        # Predict
        y_pred = model.predict(X_test_processed)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        
        print(f"{name}:")
        print(f"  MAE:  {mae:,.2f}")
        print(f"  MSE:  {mse:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  R2:   {r2:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
            
    print(f"Best Model: {best_model_name} with R2 Score: {best_r2:.4f}")
    
    # Save the best model
    best_model_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved successfully to {best_model_path}")

if __name__ == "__main__":
    train_and_evaluate_models()
