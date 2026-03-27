import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Churn AI Predictor Pro", layout="wide")


@st.cache_resource
def load_assets():
    model_path = 'churn_model.pkl'
    # Check file present or not
    if not os.path.exists(model_path):
        return None
    
    # if file is present so the file is run
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_assets()

# --- APP UI ---
st.title("🚀 Customer churn Prediction Tool")
st.markdown("---")

if model is None:
    st.error("⚠️ Model file 'churn_model.pkl' not found. first save your model.")
else:
    # Sidebar Information
    st.sidebar.header("Instructions")
    st.sidebar.info(
        "For the churn prediction please upload the csv which contains 15 columns that we were used in training.")

    # Tabs for Single vs Batch
    tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📂 churn CSV Prediction"])

    # --- TAB 1: SINGLE PREDICTION  
    with tab1:
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("Age", 18, 100, 30)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            city = st.number_input("City ID", value=1)
            unique_cat = st.slider("Unique Categories", 1, 50, 5)
            total_spend = st.number_input("Total Spend ($)", value=500.0)
        with c2:
            avg_order = st.number_input("Avg Order Value", value=100.0)
            total_qty = st.number_input("Total Quantity", value=10)
            total_disc = st.number_input("Total Discount", value=20.0)
            total_orders = st.number_input("Total Orders", value=5)
            avg_session = st.number_input("Avg Session (min)", value=15.0)
        with c3:
            avg_pages = st.number_input("Pages Viewed", value=10)
            avg_delivery = st.number_input("Delivery Days", value=3)
            avg_rating = st.slider("Rating", 1.0, 5.0, 4.0)
            recency = st.number_input("Recency", value=30)
            returning = st.radio("Returning?", [0, 1])

        if st.button("Predict Single"):
            features = [age, gender, city, unique_cat, total_spend, avg_order, total_qty,
                        total_disc, total_orders, avg_session, avg_pages, avg_delivery,
                        avg_rating, recency, returning]
            input_df = pd.DataFrame([features], columns=model.get_booster().feature_names)
            prob = model.predict_proba(input_df)[0][1]
            if prob > 0.5:
                st.error(f"High Risk: {prob * 100:.1f}%")
            else:
                st.success(f"Low Risk: {(1 - prob) * 100:.1f}%")

    # --- TAB 2: BATCH PREDICTION --
    with tab2:
        st.subheader("Upload CSV for Bulk Analysis")
        uploaded_file = st.file_uploader("CSV file yahan drop karein", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("File Preview (Pehli 5 rows):")
            st.dataframe(data.head())

            if st.button("Process Batch Prediction"):
                try:
                    
                    model_features = model.get_booster().feature_names
                    
                    missing = [c for c in model_features if c not in data.columns]

                    if missing:
                        st.error(f"Missing Columns: {missing}")
                    else:
                        # Prediction 
                        predictions = model.predict(data[model_features])
                        probabilities = model.predict_proba(data[model_features])[:, 1]

                        # Add result in original dataset
                        data['Churn_Prediction'] = predictions
                        data['Churn_Probability'] = probabilities
                        data['Status'] = data['Churn_Prediction'].map({1: '⚠️ At Risk', 0: '✅ Safe'})

                        st.success("Analysis Complete!")
                        st.dataframe(data[['Customer_ID', 'Status', 'Churn_Probability']].head(
                            10) if 'Customer_ID' in data.columns else data.head(10))

                        # Download Button
                        csv = data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Processed Results 📥",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.caption("Developed by Usman Waris | Data Scientist")
