import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

def load_and_clean_data(data_path='data/housing.csv'):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Remove duplicates
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    # 2. Treat Outliers using IQR method (for Area_sqft and Price)
    for col in ['Area_sqft', 'Price']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        num_outliers = outliers_mask.sum()
        
        # Cap outliers instead of dropping them so we don't lose too much data
        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound
        print(f"Capped {num_outliers} outliers in {col}.")
    
    return df

def preprocess_and_save(df, models_dir='models', clean_data_path='data/cleaned_housing.csv'):
    os.makedirs(models_dir, exist_ok=True)
    
    # Separate features and target
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Identify column types
    numeric_features = ['Bedrooms', 'Area_sqft', 'Bathrooms', 'Parking_spaces', 'House_age']
    categorical_features = ['Location']
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit preprocessor on X
    preprocessor.fit(X)
    
    # Save the preprocessor
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
    print("Preprocessor saved successfully.")
    
    # We will also save the cleaned dataset (before scaling/encoding) for the modeling script to split
    df.to_csv(clean_data_path, index=False)
    print(f"Cleaned dataset saved to {clean_data_path}")
    
    return preprocessor

if __name__ == "__main__":
    df = load_and_clean_data()
    preprocess_and_save(df)
