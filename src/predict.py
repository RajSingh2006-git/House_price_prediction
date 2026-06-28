import pandas as pd
import joblib
import os

def predict_price(bedrooms, area_sqft, bathrooms, location, parking_spaces, house_age, 
                  models_dir='models'):
    """
    Predict the house price using the trained model and preprocessor.
    """
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    model_path = os.path.join(models_dir, 'best_model.joblib')
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or preprocessor not found. Please run the training pipeline first.")
        
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Bedrooms': [bedrooms],
        'Area_sqft': [area_sqft],
        'Bathrooms': [bathrooms],
        'Location': [location],
        'Parking_spaces': [parking_spaces],
        'House_age': [house_age]
    })
    
    # Preprocess
    input_processed = preprocessor.transform(input_data)
    
    # Predict
    prediction = model.predict(input_processed)[0]
    
    return prediction

if __name__ == "__main__":
    # Test the prediction function
    price = predict_price(3, 2000, 2, 'Urban', 1, 10)
    print(f"Predicted Price: ${price:,.2f}")
