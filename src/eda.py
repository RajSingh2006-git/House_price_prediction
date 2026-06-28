import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(data_path='data/housing.csv', output_dir='visualizations'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path} for EDA...")
    df = pd.read_csv(data_path)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution of House Prices
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], bins=50, kde=True, color='blue')
    plt.title('Distribution of House Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'price_distribution.png'))
    plt.close()
    
    # 2. Correlation Heatmap (excluding categorical 'Location' for correlation)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Feature Relationships with Price
    # Price vs Area
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Area_sqft', y='Price', data=df, hue='Location', alpha=0.6)
    plt.title('Price vs Area (sqft)')
    plt.savefig(os.path.join(output_dir, 'price_vs_area.png'))
    plt.close()
    
    # Price vs Bedrooms
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Bedrooms', y='Price', data=df)
    plt.title('Price vs Bedrooms')
    plt.savefig(os.path.join(output_dir, 'price_vs_bedrooms.png'))
    plt.close()

    # Price vs Location
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Location', y='Price', data=df)
    plt.title('Price vs Location')
    plt.savefig(os.path.join(output_dir, 'price_vs_location.png'))
    plt.close()
    
    print(f"EDA visualizations saved to {output_dir}/")

if __name__ == "__main__":
    perform_eda()
