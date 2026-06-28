import pandas as pd
import numpy as np
import os

def generate_housing_data(num_samples=2000, output_path='data/housing.csv'):
    np.random.seed(42)
    
    bedrooms = np.random.randint(1, 7, size=num_samples)
    area = np.random.uniform(500, 5000, size=num_samples)
    bathrooms = np.random.randint(1, 6, size=num_samples)
    locations = np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_samples)
    parking_spaces = np.random.randint(0, 4, size=num_samples)
    house_age = np.random.randint(0, 100, size=num_samples)
    
    # Base price calculation
    price = (area * 150) + (bedrooms * 20000) + (bathrooms * 15000) + (parking_spaces * 10000) - (house_age * 1000)
    
    # Location multipliers
    loc_mult = {'Urban': 1.5, 'Suburban': 1.2, 'Rural': 0.8}
    price = price * np.array([loc_mult[loc] for loc in locations])
    
    # Add some random noise
    noise = np.random.normal(0, 50000, size=num_samples)
    price = price + noise
    price = np.maximum(price, 50000) # Minimum price
    
    df = pd.DataFrame({
        'Bedrooms': bedrooms,
        'Area_sqft': area,
        'Bathrooms': bathrooms,
        'Location': locations,
        'Parking_spaces': parking_spaces,
        'House_age': house_age,
        'Price': price
    })
    
    # Introduce missing values (approx 5% per column)
    for col in df.columns:
        if col != 'Price':
            mask = np.random.rand(num_samples) < 0.05
            df.loc[mask, col] = np.nan
            
    # Introduce duplicates (approx 2% of data)
    num_duplicates = int(num_samples * 0.02)
    indices_to_duplicate = np.random.choice(df.index, size=num_duplicates, replace=False)
    duplicates = df.loc[indices_to_duplicate]
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Introduce outliers in Area_sqft and Price
    outlier_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_idx, 'Area_sqft'] = df.loc[outlier_idx, 'Area_sqft'] * 3
    df.loc[outlier_idx, 'Price'] = df.loc[outlier_idx, 'Price'] * 3
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    generate_housing_data()
