import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD ARTIFACTS ----------------
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
SCALAR_PATH = ARTIFACT_DIR / "scalar.pkl"
META_PATH = ARTIFACT_DIR / "meta.pkl"

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALAR_PATH, 'rb') as f:
        scalar = pickle.load(f)
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    return model, scalar, meta

model, scalar, meta = load_artifacts()

# ---------------- UTILS ----------------
def make_feature_vector(input_dict, scalar, meta):
    """
    Create a feature vector aligned to training columns, including Brand dummies,
    then apply the fitted StandardScaler.
    """
    df = pd.DataFrame([input_dict])
    
    # One-hot encode Brand
    df_d = pd.get_dummies(df, columns=['Brand'], prefix='Brand')
    
    # Ensure all expected Brand columns exist
    for bcol in meta['expected_brand_cols']:
        if bcol not in df_d.columns:
            df_d[bcol] = 0
    
    # Ensure final column order matches training
    for col in meta['feature_order']:
        if col not in df_d.columns:
            df_d[col] = 0.0
    df_d = df_d[meta['feature_order']]
    
    # Apply StandardScaler (fitted on all features including dummy columns)
    df_scaled = pd.DataFrame(scalar.transform(df_d), columns=df_d.columns)
    
    return df_scaled

def format_currency(x):
    return f"₹ {x:,.2f}"

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;color:#4B7BEC;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Predict laptop prices quickly based on your configuration</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- INPUT FORM ----------------
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", options=['Dell','HP','Lenovo','Asus','Acer'])
        processor_speed = st.slider("Processor Speed (GHz)", 1.5, 4.0, 2.6, 0.1)
        ram_map = {'4 GB':4, '8 GB':8, '16 GB':16, '32 GB':32}
        ram_size = ram_map[st.selectbox("RAM Size", options=list(ram_map.keys()), index=1)]
    with col2:
        storage_map = {'256 GB':256, '512 GB':512, '1 TB (1000 GB)':1000}
        storage_capacity = storage_map[st.selectbox("Storage Capacity", options=list(storage_map.keys()), index=1)]
        screen_size = st.slider("Screen Size (inches)", 11.0, 17.0, 15.6, 0.1)
    
    submitted = st.form_submit_button("Predict Price 💸")

# ---------------- PREDICTION ----------------
if submitted:
    input_dict = {
        'Brand': brand,
        'Processor_Speed': float(processor_speed),
        'RAM_Size': int(ram_size),
        'Storage_Capacity': int(storage_capacity),
        'Screen_Size': float(screen_size)
    }
    
    X_input = make_feature_vector(input_dict, scalar, meta)
    pred = model.predict(X_input)[0]
    
    # Estimated range
    margin = max(50, 0.05 * abs(pred))
    lower, upper = pred - margin, pred + margin
    
    # Display prediction
    st.markdown(
        f"<div style='background:linear-gradient(90deg,#f0f9ff,#e9f7ef);padding:20px;border-radius:12px'>"
        f"<h2 style='color:#1F77B4;'>Predicted Price: {format_currency(pred)}</h2>"
        f"<p>Estimated Range: {format_currency(lower)} - {format_currency(upper)}</p>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Show top influential features
    coef = model.coef_
    feat_names = meta['feature_order']
    coef_df = pd.DataFrame({'feature': feat_names, 'coefficient': coef})
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    top = coef_df.sort_values('abs_coef', ascending=False).head(8)[['feature','coefficient']]
    
    st.subheader("Top Features Influencing Price")
    st.table(top.set_index('feature'))
    
    # Download CSV
    result_df = X_input.copy()
    result_df['predicted_price'] = pred
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Prediction CSV", csv, file_name="prediction.csv", mime="text/csv")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#888;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
