import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ==============================
# App Config
# ==============================
st.set_page_config(
    page_title="✈️ Airline Passenger Satisfaction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_airline_passenger_satisfaction.csv")

df = load_data()

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("best_random_forest_pipeline.joblib")

model = load_model()

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("📑 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Overview of the Data", "EDA", "Model Prediction"])

# ==============================
# Home Page
# ==============================
if page == "Home":
    st.title("✈️ Airline Passenger Satisfaction Dashboard")
    st.markdown("""
    Welcome to the **Airline Passenger Satisfaction App**! 🎉  
    Here you can:
    - 📊 Explore passenger survey data  
    - 🔎 Discover patterns with interactive visualizations  
    - 🤖 Test a machine learning model that predicts passenger satisfaction  

    Use the sidebar to navigate through the app 👉
    """)
    st.image("data/medium-large.jpg")

# ==============================
# Overview of the Data
# ==============================
elif page == "Overview of the Data":
    st.header("📋 Dataset Overview")
    
    st.write("This dataset contains passenger demographics, flight details, and survey responses about in-flight experience.")
    
    st.subheader("Preview of the Dataset")
    st.dataframe(df.head())

    st.subheader("Data Info")
    st.write(f"Shape of dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Summary Statistics")
    st.write(df.describe())

# ==============================
# EDA Page
# ==============================
elif page == "EDA":
    st.header("🔍 Exploratory Data Analysis")

    st.subheader("Distribution of Passenger Satisfaction")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.countplot(x="satisfaction", data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

    st.subheader("Age Distribution of Passengers")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax, color="skyblue")
    st.pyplot(fig)

    st.subheader("Flight Distance by Class")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x="Class", y="Flight Distance", data=df, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Average Satisfaction by Inflight WiFi Service")
    avg_wifi = df.groupby("Inflight wifi service")["satisfaction"].value_counts(normalize=True).unstack().fillna(0)
    st.bar_chart(avg_wifi)

# ==============================
# Model Prediction Page
# ==============================
elif page == "Model Prediction":
    st.header("🤖 Passenger Satisfaction Prediction")

    st.markdown("Fill in the passenger details below and click **Predict** to see if they are satisfied or not.")

    # Collect user inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    flight_distance = st.number_input("Flight Distance", min_value=50, max_value=5000, value=500)
    wifi = st.slider("Inflight WiFi Service", 0, 5, 3)
    comfort = st.slider("Seat Comfort", 0, 5, 3)
    entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 4)

    # Create raw dataframe
    input_df = pd.DataFrame({
        "Gender": [gender],
        "Customer Type": [customer_type],
        "Type of Travel": [travel_type],
        "Class": [travel_class],
        "Age": [age],
        "Flight Distance": [flight_distance],
        "Inflight wifi service": [wifi],
        "Seat comfort": [comfort],
        "Inflight entertainment": [entertainment],
        "Cleanliness": [cleanliness]
    })

    # Load model and feature names
    model, feature_names = joblib.load("rf_model.pkl")

    # One-hot encode user input
    input_encoded = pd.get_dummies(input_df)

    # Align with training features
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    # Debugging (optional, remove later)
    st.write("Model expects:", model.feature_names_in_)
    st.write("Input columns:", input_encoded.columns.tolist())

    # Prediction
    if st.button("🔮 Predict Satisfaction"):
        prediction = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        if prediction == "satisfied":
            st.success(f"✅ The passenger is predicted to be **SATISFIED** (Confidence: {prob:.2%})")
        else:
            st.error(f"❌ The passenger is predicted to be **DISSATISFIED** (Confidence: {1-prob:.2%})")
