import streamlit as st
import os
from PIL import Image
from src.predict import predict_price

# Set page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 Machine Learning House Price Predictor")
st.markdown("Predict the estimated house price based on various features using a trained Machine Learning model.")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Price", "Exploratory Data Analysis"])

if page == "Predict Price":
    st.header("Enter House Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
        
    with col2:
        area_sqft = st.number_input("Area (sq ft)", min_value=100.0, max_value=20000.0, value=1500.0, step=50.0)
        parking_spaces = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1, step=1)
        house_age = st.number_input("House Age (years)", min_value=0, max_value=200, value=10, step=1)
        
    if st.button("Predict Price", type="primary"):
        with st.spinner("Calculating..."):
            try:
                price = predict_price(
                    bedrooms=bedrooms,
                    area_sqft=area_sqft,
                    bathrooms=bathrooms,
                    location=location,
                    parking_spaces=parking_spaces,
                    house_age=house_age
                )
                st.success(f"### Estimated House Price: ${price:,.2f}")
            except FileNotFoundError:
                st.error("Model files not found! Please ensure you have run the training pipeline.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif page == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Visualizations generated from the housing dataset during the EDA phase.")
    
    vis_dir = 'visualizations'
    
    if os.path.exists(vis_dir):
        # Distribution
        st.subheader("Distribution of House Prices")
        if os.path.exists(os.path.join(vis_dir, 'price_distribution.png')):
            st.image(os.path.join(vis_dir, 'price_distribution.png'), use_container_width=True)
            
        # Correlation
        st.subheader("Feature Correlation Heatmap")
        if os.path.exists(os.path.join(vis_dir, 'correlation_heatmap.png')):
            st.image(os.path.join(vis_dir, 'correlation_heatmap.png'), use_container_width=True)
            
        # Relationships
        st.subheader("Price Relationships")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(os.path.join(vis_dir, 'price_vs_area.png')):
                st.image(os.path.join(vis_dir, 'price_vs_area.png'), caption="Price vs Area (sqft)", use_container_width=True)
        with col2:
            if os.path.exists(os.path.join(vis_dir, 'price_vs_bedrooms.png')):
                st.image(os.path.join(vis_dir, 'price_vs_bedrooms.png'), caption="Price vs Bedrooms", use_container_width=True)
                
        if os.path.exists(os.path.join(vis_dir, 'price_vs_location.png')):
            st.image(os.path.join(vis_dir, 'price_vs_location.png'), caption="Price vs Location", use_container_width=True)
    else:
        st.info("Visualizations not found. Run the EDA script to generate them.")

st.markdown("---")
st.markdown("Built with ❤️ using Python, Scikit-Learn, XGBoost, and Streamlit.")
