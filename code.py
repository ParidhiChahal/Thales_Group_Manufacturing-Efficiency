import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Manufacturing Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Thales_Group_Manufacturing.csv')
    
    # Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Use lowercase names
if 'date' in df.columns and 'time' in df.columns:
    df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    return df

df = load_data()

st.title("AI-Powered Manufacturing Efficiency Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
machine = st.sidebar.selectbox("Select Machine", df['Machine_ID'].unique())
mode = st.sidebar.selectbox("Operation Mode", df['Operation_Mode'].unique())

start_date = st.sidebar.date_input("Start Date", df['Datetime'].min().date())
end_date = st.sidebar.date_input("End Date", df['Datetime'].max().date())

filtered_df = df[
    (df['Machine_ID'] == machine) &
    (df['Operation_Mode'] == mode) &
    (df['Datetime'].dt.date >= start_date) &
    (df['Datetime'].dt.date <= end_date)
]

# KPIs
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Production", int(filtered_df['Production_Speed_units_per_hr'].mean()))
col2.metric("Avg Error Rate", round(filtered_df['Error_Rate_%'].mean(), 2))
col3.metric("Avg Power", round(filtered_df['Power_Consumption_kW'].mean(), 2))

# Preprocessing
le_mode = LabelEncoder()
le_target = LabelEncoder()

df['Operation_Mode'] = le_mode.fit_transform(df['Operation_Mode'])
df['Efficiency_Status'] = le_target.fit_transform(df['Efficiency_Status'])

features = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
            'Network_Latency_ms', 'Packet_Loss_%',
            'Quality_Control_Defect_Rate_%',
            'Production_Speed_units_per_hr',
            'Predictive_Maintenance_Score', 'Error_Rate_%']

X = df[features]
y = df['Efficiency_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Real-time prediction inputs
st.subheader("⚡ Real-Time Prediction")
input_values = []
for feature in features:
    val = st.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    input_values.append(val)

input_array = np.array(input_values).reshape(1, -1)
input_scaled = scaler.transform(input_array)

pred = model.predict(input_scaled)[0]
confidence = np.max(model.predict_proba(input_scaled))

label_map = {i: label for i, label in enumerate(le_target.classes_)}

st.success(f"Predicted Efficiency: {label_map[pred]}")
st.info(f"Confidence: {round(confidence * 100, 2)}%")

# Feature Importance
st.subheader("📌 Feature Importance")
importance = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(features, importance)
st.pyplot(fig)

# Trend Analysis
st.subheader("📈 Production Trend")
fig2, ax2 = plt.subplots()
ax2.plot(filtered_df['Datetime'], filtered_df['Production_Speed_units_per_hr'])
ax2.set_xlabel("Time")
ax2.set_ylabel("Production Speed")
st.pyplot(fig2)

# Efficiency Distribution
st.subheader("📊 Efficiency Distribution")
fig3, ax3 = plt.subplots()
ax3.hist(df['Efficiency_Status'])
st.pyplot(fig3)

st.markdown("---")
st.caption("Developed for Smart Manufacturing AI Project")
