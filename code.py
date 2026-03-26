import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Manufacturing Dashboard", layout="wide")

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv('Thales_Group_Manufacturing.csv')
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    
    # Create Datetime safely
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    return df

# Load dataset
df = load_data()

# ===================== SIDEBAR =====================
st.sidebar.header("Filters")

if 'Machine_ID' in df.columns:
    machine = st.sidebar.selectbox("Select Machine", df['Machine_ID'].unique())
else:
    st.error("Machine_ID column not found")
    st.stop()

if 'Operation_Mode' in df.columns:
    mode = st.sidebar.selectbox("Operation Mode", df['Operation_Mode'].unique())
else:
    mode = None

# Filter data
filtered_df = df[df['Machine_ID'] == machine]
if mode:
    filtered_df = filtered_df[filtered_df['Operation_Mode'] == mode]

# ===================== KPIs =====================
st.title(" AI-Powered Manufacturing Efficiency Dashboard")

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Avg Production", int(filtered_df['Production_Speed_units_per_hr'].mean()))
col2.metric("Avg Error Rate", round(filtered_df['Error_Rate_%'].mean(), 2))
col3.metric("Avg Power", round(filtered_df['Power_Consumption_kW'].mean(), 2))

# ===================== PREPROCESSING =====================
df = df.dropna()

le = LabelEncoder()

if 'Operation_Mode' in df.columns:
    df['Operation_Mode'] = le.fit_transform(df['Operation_Mode'])

if 'Efficiency_Status' in df.columns:
    df['Efficiency_Status'] = le.fit_transform(df['Efficiency_Status'])

features = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
            'Network_Latency_ms', 'Packet_Loss_%',
            'Quality_Control_Defect_Rate_%',
            'Production_Speed_units_per_hr',
            'Predictive_Maintenance_Score', 'Error_Rate_%']

# Keep only available columns
features = [col for col in features if col in df.columns]

X = df[features]
y = df['Efficiency_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===================== MODEL =====================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ===================== REAL-TIME PREDICTION =====================
st.subheader(" Real-Time Prediction")

input_values = []
for feature in features:
    val = st.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    input_values.append(val)

input_array = np.array(input_values).reshape(1, -1)
input_scaled = scaler.transform(input_array)

pred = model.predict(input_scaled)[0]
confidence = np.max(model.predict_proba(input_scaled))

st.success(f"Predicted Efficiency: {pred}")
st.info(f"Confidence: {round(confidence * 100, 2)}%")

# ===================== FEATURE IMPORTANCE =====================
st.subheader(" Feature Importance")
importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importance)
st.pyplot(fig)

# ===================== TREND =====================
if 'Datetime' in filtered_df.columns:
    st.subheader(" Production Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(filtered_df['Datetime'], filtered_df['Production_Speed_units_per_hr'])
    st.pyplot(fig2)

st.markdown("---")
st.caption("Developed for Smart Manufacturing AI Project")
