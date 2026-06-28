# 🏠 Machine Learning House Price Predictor

Welcome to the House Price Predictor project! This is a complete end-to-end Machine Learning pipeline and web application built with Python, Scikit-Learn, XGBoost, and Streamlit. 

## 🚀 Overview

This repository demonstrates a full ML lifecycle, including:
1. **Data Generation**: A synthetic dataset creator that models realistic house prices based on various features.
2. **Exploratory Data Analysis (EDA)**: Visualizations to understand data distributions and feature correlations.
3. **Data Preprocessing**: Handling missing values, capping outliers (IQR method), scaling numeric features, and encoding categorical data seamlessly using Scikit-Learn pipelines.
4. **Model Training & Evaluation**: Training multiple regression models (Linear Regression, Decision Tree, Random Forest, XGBoost) and selecting the best one (Random Forest) based on the R² score.
5. **Interactive Web App**: A beautiful Streamlit interface for users to enter house details and instantly get a price prediction.

## 📁 Project Structure

```text
house_price_prediction/
├── data/
│   ├── housing.csv               # Raw synthetic dataset
│   └── cleaned_housing.csv       # Dataset after outlier treatment
├── models/
│   ├── best_model.joblib         # Serialized Random Forest model
│   └── preprocessor.joblib       # Serialized data preprocessor pipeline
├── src/
│   ├── data_generation.py        # Generates the synthetic dataset
│   ├── eda.py                    # Generates plots from the data
│   ├── preprocessing.py          # Cleans and scales data
│   ├── modeling.py               # Trains models and evaluates them
│   └── predict.py                # Core prediction logic
├── visualizations/               # Saved charts from EDA (heatmaps, distributions)
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation (You are here!)
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RajSingh2006-git/House_price_prediction.git
   cd House_price_prediction
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Running the Pipeline

If you want to re-run the training pipeline from scratch:

1. **Generate the data:**
   ```bash
   python src/data_generation.py
   ```
2. **Perform Exploratory Data Analysis:**
   ```bash
   python src/eda.py
   ```
3. **Run preprocessing:**
   ```bash
   python src/preprocessing.py
   ```
4. **Train the models:**
   ```bash
   python src/modeling.py
   ```

*(Note: The repository already contains the trained `.joblib` model and preprocessor, so you can skip this step if you just want to run the web app!)*

## 🌐 Running the Web Application

To launch the interactive Streamlit UI, simply run:
```bash
streamlit run app.py
```
This will open the app in your default web browser where you can input parameters like **Bedrooms, Bathrooms, Area, Location**, etc., and get an estimated house price prediction in real-time.

## 📈 Performance

During evaluation, the **Random Forest Regressor** was automatically selected as the best performing model, achieving an **R² Score of ~0.90**. 

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
